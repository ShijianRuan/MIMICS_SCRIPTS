#!/usr/bin/env python3
"""Failure-path tests for the SLIP worker's publish reconciliation.

slip_bridge.py runs under the external Python (3.13), and the heavy parts
(model load, CUDA) are behind _SlipSession.open().  These tests construct
the session directly, attach a deterministic fake of the vendored
SLIP Inference_module and inject publish failures via _write_mask_u8, so
they need no GPU and never touch the vendored tree.  Covered failure paths
(review a621093 / R91-4 / SL04):

- predict succeeds but publishing the mask file fails: the retry of the
  same click republishes the cached mask (no double inference, history
  stays at 1), a different click drops the orphaned prompt first, an undo
  then targets the prompt the client actually saw;
- undo succeeds but publishing fails: the next undo republishes the cached
  post-undo mask, the next predict replays the undone click so the worker
  history matches what the client counted;
- an exception inside the module call poisons the session: further
  predict/undo/reset commands are refused instead of building on state
  that cannot be trusted;
- reset absorbs any half-published change;
- a publish that fails twice keeps the pending state consistent.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import slip_bridge  # noqa: E402


SHAPE = (3, 4, 5)


class FakeTensor:
    """Minimal stand-in for the torch tensor the bridge expects back."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=bool)

    def squeeze(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def astype(self, dtype):
        return self._array.astype(dtype)


class FakeSlipModule:
    """Deterministic fake of the vendored SLIP Inference_module.

    Mirrors only the semantics the bridge relies on:

    - click_inference appends {"name", "coords", "type"} to action_history
      and returns a mask whose voxel count identifies the prompt count;
    - undo pops the newest history entry in place, or fully resets when it
      is the only one (stack semantics of the vendored implementation);
    - reset clears history and patch bookkeeping.

    ``calls`` records every invocation so tests can assert exactly how many
    inferences really ran.
    """

    def __init__(self, shape):
        self.action_history = []
        self.bbox_data = {"patch-0": object()}
        self._shape = shape
        self.calls = []
        self.fail_inference = False

    def _mask_for(self, count):
        array = np.zeros(self._shape, dtype=bool)
        if count:
            array.reshape(-1)[: count * 3] = True
        return FakeTensor(array)

    def click_inference(self, clicks, labels):
        self.calls.append(("predict", list(clicks[0]), labels[0]))
        if self.fail_inference:
            raise RuntimeError("inference exploded mid-way")
        click = [int(value) for value in clicks[0]]
        self.action_history.append({
            "name": len(self.action_history) + 1,
            "coords": [click],
            "type": "include" if labels[0] == 1 else "exclude",
        })
        return self._mask_for(len(self.action_history))

    def undo(self, action_history):
        self.calls.append(("undo",))
        if len(action_history) > 1:
            action_history.pop()
        else:
            # reset() rebinds the attribute, so read the live history back
            # instead of the (possibly detached) parameter.
            self.reset()
        return self._mask_for(len(self.action_history))

    def reset(self):
        self.calls.append(("reset",))
        self.action_history = []
        self.bbox_data = {}


class PublishFailure:
    """Context manager replacing slip_bridge._write_mask_u8 with a raiser."""

    def __enter__(self):
        self._original = slip_bridge._write_mask_u8
        slip_bridge._write_mask_u8 = self._fail
        return self

    def __exit__(self, *exc_info):
        slip_bridge._write_mask_u8 = self._original
        return False

    @staticmethod
    def _fail(mask, output_path):
        raise OSError("disk full while publishing mask")


class SlipBridgeReconcileTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="slip_bridge_test_")
        self.addCleanup(self._tmp.cleanup)
        self.output_path = os.path.join(self._tmp.name, "out.u8")

    def _session(self):
        session = slip_bridge._SlipSession({})
        session.module = FakeSlipModule(SHAPE)
        session.image_shape = SHAPE
        return session

    def _read_output(self):
        with open(self.output_path, "rb") as handle:
            data = handle.read()
        array = np.frombuffer(data, dtype=np.uint8).reshape(SHAPE)
        return array.astype(bool)

    # -- predict publish failure -------------------------------------------

    def test_predict_publish_failure_same_click_retry_keeps_history(self):
        session = self._session()
        module = session.module
        with PublishFailure():
            with self.assertRaises(OSError):
                session.predict([1, 1, 1], 1, self.output_path)
        # The inference itself committed on the worker side ...
        self.assertEqual(len(module.action_history), 1)
        self.assertEqual(module.calls, [("predict", [1, 1, 1], 1)])

        result = session.predict([1, 1, 1], 1, self.output_path)
        # ... and the retry only republished it: still one inference.
        self.assertEqual(result["action_history_len"], 1)
        self.assertEqual(module.calls, [("predict", [1, 1, 1], 1)])
        self.assertEqual(np.count_nonzero(self._read_output()), 3)

        # One undo returns to the pre-click state (the fake records the
        # internal reset the vendored undo performs on its last entry).
        undone = session.undo(self.output_path)
        self.assertEqual(undone["action_history_len"], 0)
        self.assertEqual(
            module.calls,
            [("predict", [1, 1, 1], 1), ("undo",), ("reset",)])
        self.assertEqual(np.count_nonzero(self._read_output()), 0)

    def test_predict_publish_failure_different_click_drops_orphan(self):
        session = self._session()
        module = session.module
        with PublishFailure():
            with self.assertRaises(OSError):
                session.predict([1, 1, 1], 1, self.output_path)

        result = session.predict([2, 2, 2], 0, self.output_path)
        # Orphan dropped, then the new prompt inferred: history is exactly
        # the prompt the client has seen.
        self.assertEqual(result["action_history_len"], 1)
        self.assertEqual(
            module.calls,
            [("predict", [1, 1, 1], 1), ("undo",), ("reset",),
             ("predict", [2, 2, 2], 0)])
        self.assertEqual(module.action_history[-1]["coords"], [[2, 2, 2]])
        self.assertIsNone(session._unpublished)

    def test_predict_publish_failure_then_undo_targets_client_prompt(self):
        session = self._session()
        module = session.module
        session.predict([1, 1, 1], 1, self.output_path)  # client saw this
        with PublishFailure():
            with self.assertRaises(OSError):
                session.predict([2, 2, 2], 1, self.output_path)  # orphan

        result = session.undo(self.output_path)
        # The orphan was dropped first, so the undo removed the prompt the
        # client actually counted -> back to zero.
        self.assertEqual(result["action_history_len"], 0)
        self.assertEqual(
            module.calls,
            [("predict", [1, 1, 1], 1), ("predict", [2, 2, 2], 1),
             ("undo",), ("undo",), ("reset",)])
        self.assertEqual(np.count_nonzero(self._read_output()), 0)

    def test_predict_publish_failure_twice_keeps_pending(self):
        session = self._session()
        module = session.module
        with PublishFailure():
            for _ in range(2):
                with self.assertRaises(OSError):
                    session.predict([1, 1, 1], 1, self.output_path)
        self.assertEqual(len(module.action_history), 1)
        self.assertEqual(module.calls, [("predict", [1, 1, 1], 1)])

        result = session.predict([1, 1, 1], 1, self.output_path)
        self.assertEqual(result["action_history_len"], 1)
        self.assertEqual(module.calls, [("predict", [1, 1, 1], 1)])

    # -- undo publish failure ------------------------------------------------

    def test_undo_publish_failure_next_undo_republishes_once(self):
        session = self._session()
        module = session.module
        session.predict([1, 1, 1], 1, self.output_path)
        session.predict([1, 2, 3], 1, self.output_path)
        with PublishFailure():
            with self.assertRaises(OSError):
                session.undo(self.output_path)
        # Worker history already decremented to 1.
        self.assertEqual(len(module.action_history), 1)

        result = session.undo(self.output_path)
        # Republished, not undone a second time.
        self.assertEqual(result["action_history_len"], 1)
        self.assertEqual(
            module.calls,
            [("predict", [1, 1, 1], 1), ("predict", [1, 2, 3], 1),
             ("undo",)])
        self.assertIsNone(session._unpublished)

    def test_undo_publish_failure_next_predict_replays_click(self):
        session = self._session()
        module = session.module
        session.predict([1, 1, 1], 1, self.output_path)  # client count: 1
        with PublishFailure():
            with self.assertRaises(OSError):
                session.undo(self.output_path)  # worker count: 0
        self.assertEqual(len(module.action_history), 0)

        result = session.predict([2, 2, 2], 1, self.output_path)
        # The undone click is replayed so the worker history matches the
        # client's count, then the new prompt is inferred on top.
        self.assertEqual(result["action_history_len"], 2)
        self.assertEqual(
            module.calls,
            [("predict", [1, 1, 1], 1), ("undo",), ("reset",),
             ("predict", [1, 1, 1], 1), ("predict", [2, 2, 2], 1)])
        self.assertEqual(module.action_history[-1]["coords"], [[2, 2, 2]])

    def test_undo_of_only_prompt_publish_failure_replays_it(self):
        session = self._session()
        module = session.module
        session.predict([0, 1, 2], 0, self.output_path)
        with PublishFailure():
            with self.assertRaises(OSError):
                session.undo(self.output_path)
        self.assertEqual(len(module.action_history), 0)

        result = session.predict([2, 0, 1], 1, self.output_path)
        # Replay must restore the original include/exclude label, not
        # silently turn the prompt into a positive one.
        self.assertEqual(result["action_history_len"], 2)
        self.assertEqual(
            module.calls,
            [("predict", [0, 1, 2], 0), ("undo",), ("reset",),
             ("predict", [0, 1, 2], 0), ("predict", [2, 0, 1], 1)])
        self.assertEqual(module.action_history[0]["type"], "exclude")

    # -- poisoning -------------------------------------------------------------

    def test_inference_exception_poisons_session(self):
        session = self._session()
        module = session.module
        module.fail_inference = True
        with self.assertRaises(RuntimeError):
            session.predict([1, 1, 1], 1, self.output_path)
        calls_after_failure = list(module.calls)

        for command in (
            lambda: session.predict([2, 2, 2], 1, self.output_path),
            lambda: session.undo(self.output_path),
            session.reset,
        ):
            with self.assertRaises(RuntimeError) as caught:
                command()
            self.assertIn("no longer be trusted", str(caught.exception))
        # No further module calls were attempted after the failure.
        self.assertEqual(module.calls, calls_after_failure)

    def test_validation_error_does_not_poison(self):
        session = self._session()
        module = session.module
        with self.assertRaises(RuntimeError):
            session.predict([99, 0, 0], 1, self.output_path)  # out of range
        self.assertFalse(session._poisoned)
        session.predict([1, 1, 1], 1, self.output_path)
        self.assertEqual(len(module.action_history), 1)

    # -- reset ----------------------------------------------------------------

    def test_reset_absorbs_pending_orphan(self):
        session = self._session()
        module = session.module
        with PublishFailure():
            with self.assertRaises(OSError):
                session.predict([1, 1, 1], 1, self.output_path)

        result = session.reset()
        self.assertEqual(result["action_history_len"], 0)
        self.assertIsNone(session._unpublished)
        self.assertEqual(len(module.action_history), 0)

        # The session continues cleanly from a fresh history.
        after = session.predict([2, 2, 2], 1, self.output_path)
        self.assertEqual(after["action_history_len"], 1)
        self.assertEqual(module.action_history[-1]["coords"], [[2, 2, 2]])

    def test_undo_with_no_history_refuses_cleanly(self):
        session = self._session()
        with self.assertRaises(RuntimeError) as caught:
            session.undo(self.output_path)
        self.assertIn("Nothing to undo", str(caught.exception))
        self.assertFalse(session._poisoned)


if __name__ == "__main__":
    unittest.main(verbosity=2)

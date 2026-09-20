"""Prompt-order contract tests for the external nnInteractive bridge."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import nninteractive_bridge as bridge


IDENTITY_MAPPING = {
    "platform_to_mimics_axes": [0, 1, 2],
    "platform_to_mimics_flips": [False, False, False],
}


class RecordingSession:
    def __init__(self) -> None:
        self.calls = []

    def add_point_interaction(
        self, point, include_interaction, run_prediction=True
    ):
        self.calls.append(
            ("point", tuple(point), bool(include_interaction), run_prediction)
        )

    def add_scribble_interaction(
        self,
        crop,
        include_interaction,
        run_prediction=True,
        interaction_bbox=None,
    ):
        self.calls.append(
            (
                "scribble",
                bool(include_interaction),
                run_prediction,
                interaction_bbox,
            )
        )


def test_correction_point_set_runs_every_prompt_in_annotator_order():
    session = RecordingSession()
    count = bridge._apply_point_set(
        session,
        {
            "interaction_type": "point_set",
            "coordinates": "mimics",
            "points": [
                {"point": [2, 3, 4], "include_interaction": True},
                {"point": [8, 9, 10], "include_interaction": False},
                {"point": [12, 13, 14], "include_interaction": True},
            ],
        },
        mimics_shape=[20, 20, 20],
        platform_shape=[20, 20, 20],
        buffer_mapping=IDENTITY_MAPPING,
    )
    assert count == 3
    assert session.calls == [
        ("point", (2, 3, 4), True, True),
        ("point", (8, 9, 10), False, True),
        ("point", (12, 13, 14), True, True),
    ]


def test_empty_mask_first_point_set_predicts_once_after_all_points():
    session = RecordingSession()
    count = bridge._apply_point_set(
        session,
        {
            "interaction_type": "point_set",
            "prediction_policy": "initial_empty_batch",
            "coordinates": "mimics",
            "points": [
                {"point": [2, 3, 4], "include_interaction": True},
                {"point": [8, 9, 10], "include_interaction": False},
                {"point": [12, 13, 14], "include_interaction": True},
            ],
        },
        mimics_shape=[20, 20, 20],
        platform_shape=[20, 20, 20],
        buffer_mapping=IDENTITY_MAPPING,
    )
    assert count == 3
    assert session.calls == [
        ("point", (2, 3, 4), True, False),
        ("point", (8, 9, 10), False, False),
        ("point", (12, 13, 14), True, True),
    ]


def test_point_set_rejects_unknown_prediction_policy():
    session = RecordingSession()
    try:
        bridge._apply_point_set(
            session,
            {
                "interaction_type": "point_set",
                "prediction_policy": "guess",
                "points": [{"point": [1, 1, 1]}],
            },
            mimics_shape=[4, 4, 4],
            platform_shape=[4, 4, 4],
            buffer_mapping=IDENTITY_MAPPING,
        )
    except RuntimeError as exc:
        assert "prediction policy" in str(exc)
    else:
        raise AssertionError("unknown point-set policy should fail closed")


def test_point_edge_rounding_is_clamped_before_server_send():
    session = RecordingSession()
    count = bridge._apply_point_set(
        session,
        {
            "interaction_type": "point_set",
            "points": [
                {
                    "point": [-0.2, 3.8, 4.0],
                    "include_interaction": True,
                }
            ],
        },
        mimics_shape=[4, 4, 4],
        platform_shape=[4, 4, 4],
        buffer_mapping=IDENTITY_MAPPING,
    )
    assert count == 1
    assert session.calls == [("point", (0, 3, 3), True, True)]


def test_point_more_than_one_voxel_outside_fails_before_server_send():
    session = RecordingSession()
    try:
        bridge._apply_point_set(
            session,
            {
                "interaction_type": "point_set",
                "points": [{"point": [-1.01, 2, 2]}],
            },
            mimics_shape=[4, 4, 4],
            platform_shape=[4, 4, 4],
            buffer_mapping=IDENTITY_MAPPING,
        )
    except RuntimeError as exc:
        assert "outside image bounds" in str(exc)
    else:
        raise AssertionError("true out-of-bounds point should fail locally")
    assert session.calls == []


def test_server_transport_failures_are_recoverable_once():
    classify = bridge._BridgeSessionContext._server_failure_is_recoverable
    assert classify(RuntimeError("[WinError 10061] connection refused"))
    assert classify(RuntimeError("HTTP status code 500: Internal Server Error"))
    assert not classify(RuntimeError("CUDA out of memory"))
    assert not classify(
        RuntimeError("HTTP status code 500: CUDA out of memory")
    )
    assert not classify(
        IndexError("index 128 is out of bounds for axis 0 with size 128")
    )
    assert not classify(RuntimeError("Index error while applying point"))


def test_unreadable_server_command_line_needs_second_liveness_evidence():
    now = bridge.time.time()
    state = {
        "pid": 321,
        "server_url": "http://127.0.0.1:1527",
        "ownership_token": "owned-token",
        "started_at_epoch": now - 1000,
        "server_heartbeat_epoch": now - 1000,
    }
    with mock.patch.object(bridge, "_process_exists", return_value=True), \
         mock.patch.object(bridge, "_process_command_line", return_value=None), \
         mock.patch.object(bridge, "_server_health_matches_state", return_value=False):
        assert not bridge._process_matches_server(state)


def test_unreadable_server_command_line_accepts_recent_health_heartbeat():
    now = bridge.time.time()
    state = {
        "pid": 321,
        "server_url": "http://127.0.0.1:1527",
        "ownership_token": "owned-token",
        "started_at_epoch": now - 1000,
        "server_heartbeat_epoch": now,
    }
    with mock.patch.object(bridge, "_process_exists", return_value=True), \
         mock.patch.object(bridge, "_process_command_line", return_value=None), \
         mock.patch.object(bridge, "_server_health_matches_state", return_value=False):
        assert bridge._process_matches_server(state)


def test_unreadable_server_command_line_retains_lock_for_active_prediction():
    now = bridge.time.time()
    state = {
        "pid": 321,
        "server_url": "http://127.0.0.1:1527",
        "ownership_token": "owned-token",
        "started_at_epoch": now - 1000,
        "server_heartbeat_epoch": now - 1000,
        "active_operation": "prediction",
        "active_operation_pid": 654,
        "active_operation_started_at_epoch": now - 120,
        "active_operation_timeout_seconds": 1800,
        "active_operation_process_start_marker": "worker-start",
    }
    with mock.patch.object(bridge, "_process_exists", return_value=True), \
         mock.patch.object(bridge, "_process_command_line", return_value=None), \
         mock.patch.object(bridge, "_server_health_matches_state", return_value=False), \
         mock.patch.object(bridge, "resource_process_matches", return_value=True):
        assert bridge._process_matches_server(state)


def test_expired_active_prediction_does_not_keep_recycled_server_pid_alive():
    now = bridge.time.time()
    state = {
        "pid": 321,
        "server_url": "http://127.0.0.1:1527",
        "ownership_token": "owned-token",
        "started_at_epoch": now - 5000,
        "server_heartbeat_epoch": now - 5000,
        "active_operation": "prediction",
        "active_operation_pid": 654,
        "active_operation_started_at_epoch": now - 2000,
        "active_operation_timeout_seconds": 1800,
        "active_operation_process_start_marker": "worker-start",
    }
    with mock.patch.object(bridge, "_process_exists", return_value=True), \
         mock.patch.object(bridge, "_process_command_line", return_value=None), \
         mock.patch.object(bridge, "_server_health_matches_state", return_value=False), \
         mock.patch.object(bridge, "resource_process_matches", return_value=True):
        assert not bridge._process_matches_server(state)


def test_server_heartbeat_update_requires_matching_ownership_token(tmp_path):
    state_path = tmp_path / "server.json"
    bridge._write_server_state(
        state_path,
        {
            "ownership_token": "owned-token",
            "server_heartbeat_epoch": 0.0,
        },
    )
    bridge._record_server_heartbeat(state_path, "wrong-token")
    assert bridge._load_server_state(state_path)["server_heartbeat_epoch"] == 0.0
    bridge._record_server_heartbeat(state_path, "owned-token")
    assert bridge._load_server_state(state_path)["server_heartbeat_epoch"] > 0.0


def test_windows_command_line_reader_falls_back_to_wmic():
    result = type(
        "Result",
        (),
        {
            "stdout": (
                "CommandLine=python -m nnInteractive.inference.server.main "
                "--api-key owned-token\n"
            )
        },
    )()
    with mock.patch.object(
        bridge.subprocess,
        "run",
        side_effect=[FileNotFoundError("powershell missing"), result],
    ):
        command = bridge._windows_process_command_line(999999999)
    assert "nnInteractive.inference.server.main" in command
    assert "owned-token" in command


def test_prediction_restarts_owned_server_once_then_replays_request():
    context = object.__new__(bridge._BridgeSessionContext)
    context.owned_state_path = None
    context.owned_token = None
    calls = []

    def predict_impl(*_args, **_kwargs):
        calls.append("predict")
        if calls.count("predict") == 1:
            raise RuntimeError("[WinError 10061] connection refused")
        return {"status": "refined"}

    context._predict_impl = predict_impl
    context._restart_owned_server = lambda exc: calls.append(
        "restart:{}".format(exc)
    )
    result = context.predict([], "result.u8")
    assert result == {"status": "refined"}
    assert calls == [
        "predict",
        "restart:[WinError 10061] connection refused",
        "predict",
    ]


def test_prediction_does_not_restart_for_model_or_cuda_failure():
    context = object.__new__(bridge._BridgeSessionContext)
    context.owned_state_path = None
    context.owned_token = None
    context._predict_impl = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory")
    )
    context._restart_owned_server = lambda _exc: (_ for _ in ()).throw(
        AssertionError("non-transport failures must not restart the server")
    )
    try:
        context.predict([], "result.u8")
    except RuntimeError as exc:
        assert "out of memory" in str(exc)
    else:
        raise AssertionError("CUDA failure should be propagated")


def test_scribble_set_predicts_after_each_scribble():
    session = RecordingSession()
    count = bridge._apply_scribble_set(
        session,
        {
            "interaction_type": "scribble_set",
            "coordinates": "mimics",
            "scribbles": [
                {
                    "polyline_points": [[4, 4, 7], [8, 8, 7]],
                    "include_interaction": True,
                },
                {
                    "polyline_points": [[12, 12, 7], [15, 15, 7]],
                    "include_interaction": False,
                },
            ],
        },
        mimics_shape=[20, 20, 20],
        platform_shape=[20, 20, 20],
        buffer_mapping=IDENTITY_MAPPING,
    )
    assert count == 2
    assert [call[1:3] for call in session.calls] == [
        (True, True),
        (False, True),
    ]

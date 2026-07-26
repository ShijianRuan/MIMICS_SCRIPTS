import json
import os

import pytest

from nninteractive_finetune.runtime import exclusive_job_lock, write_json_atomic


def test_job_lock_rejects_live_holder(tmp_path):
    path = tmp_path / "training.lock"
    path.write_text(json.dumps({"pid": os.getpid(), "job_id": "existing"}))
    with pytest.raises(RuntimeError, match="already using"):
        with exclusive_job_lock(path, "new"):
            pass


def test_job_lock_reclaims_dead_holder_and_releases(tmp_path):
    path = tmp_path / "training.lock"
    path.write_text(json.dumps({"pid": 99999999, "job_id": "dead"}))
    with exclusive_job_lock(path, "new"):
        assert path.is_file()
    assert not path.exists()


def test_json_writer_uses_direct_write_when_replace_is_denied(tmp_path, monkeypatch):
    path = tmp_path / "status.json"

    def deny_replace(*_args):
        raise PermissionError("replace denied")

    monkeypatch.setattr(os, "replace", deny_replace)
    monkeypatch.setattr(
        "nninteractive_finetune.runtime.time.sleep", lambda _value: None
    )
    write_json_atomic(path, {"status": "training", "epoch": 2})
    assert json.loads(path.read_text())["epoch"] == 2

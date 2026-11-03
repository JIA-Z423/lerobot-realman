# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""All sockets, device responses and elapsed time are simulated."""

import importlib.util
import json
from pathlib import Path
from threading import Barrier

import pytest

from lerobot.common.realman import client as client_module

SCRIPT = Path(__file__).resolve().parents[2] / "examples/realman/watch_realman_gripper.py"
spec = importlib.util.spec_from_file_location("realman_gripper_watch", SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class FakeSocket:
    def __init__(self, positions=(0,), mode=115200, base=None, barrier=None):
        self.positions = list(positions)
        self.mode = mode
        self.base = base or {"dof": 1, "pos_low": [0], "pos_up": [1000]}
        self.barrier = barrier
        self.sent = []
        self.pending = b""
        self.closed = False

    def settimeout(self, timeout):
        pass

    def sendall(self, data):
        payload = json.loads(data)
        self.sent.append(payload)
        command = payload["command"]
        response = {"command": command}
        if command == "get_rm_plus_mode":
            response["mode"] = self.mode
        elif command == "get_rm_plus_base_info":
            response.update(self.base if "get_state" in self.base else {"base_info": self.base})
        elif command == "get_rm_plus_state_info":
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            value = self.positions.pop(0) if len(self.positions) > 1 else self.positions[0]
            if isinstance(value, Exception):
                raise value
            response.update(
                value
                if isinstance(value, dict)
                else {"state_info": {"sys_state": 0, "sys_err": 0, "dof_err": [0], "pos": [value]}}
            )
        else:
            pytest.fail(f"Unexpected command: {command}")
        self.pending += (json.dumps(response) + "\r\n").encode()

    def recv(self, size):
        chunk, self.pending = self.pending[:size], self.pending[size:]
        return chunk

    def close(self):
        self.closed = True


@pytest.fixture
def config():
    return {
        "robot": {
            "type": "double_realman_follower",
            **{
                f"{side}_arm_config": {
                    "host": side,
                    "sdk_backend": "socket_json",
                    "gripper_feedback_source": "rm_plus",
                    "initialize_gripper_route": False,
                    "initialize_gripper_modbus": False,
                    "gripper_scale": 1000.0,
                    "cameras": {"unused": {"type": "not-a-camera"}},
                }
                for side in ("left", "right")
            },
        }
    }


@pytest.fixture
def transport(monkeypatch):
    sockets = {"left": FakeSocket(), "right": FakeSocket()}
    attempts = []

    def connect(address, **kwargs):
        attempts.append(address)
        return sockets[address[0]]

    def forbidden(*args, **kwargs):
        pytest.fail("The monitor must never configure devices or send targets")

    monkeypatch.setattr(client_module.socket, "create_connection", connect)
    monkeypatch.setattr(client_module.RealManSocketClient, "configure", forbidden)
    monkeypatch.setattr(client_module.RealManSocketClient, "send_joint_positions", forbidden)
    return sockets, attempts


@pytest.fixture
def clock(monkeypatch):
    class FakeClock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    fake = FakeClock()
    monkeypatch.setattr(watch.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(watch.time, "sleep", fake.sleep)
    return fake


@pytest.mark.parametrize(
    "command", ["set_rm_plus_mode", "set_gripper_position", "movej_canfd", "get_joint_degree"]
)
def test_monitor_whitelist_rejects_other_commands_before_send(config, transport, command):
    sockets, _ = transport
    client = watch.ReadOnlyGripperClient(watch.client_configs(config)["left"])
    client.connect()
    try:
        with pytest.raises(ValueError, match="Refusing non-monitor"):
            client._send_payload({"command": command})
        assert sockets["left"].sent == []
    finally:
        client.disconnect()
    assert sockets["left"].closed


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "option,value",
    [
        ("initialize_gripper_route", True),
        ("initialize_gripper_modbus", True),
        ("gripper_feedback_source", "native"),
    ],
)
def test_unsafe_config_rejected_before_any_connection(config, transport, side, option, value):
    _, attempts = transport
    config["robot"][f"{side}_arm_config"][option] = value
    with pytest.raises(ValueError, match=option):
        watch.watch_grippers(config, duration_s=0.1)
    assert attempts == []


def test_changes_and_errors_use_production_parser_without_cached_zero(
    config, transport, clock, tmp_path, capsys
):
    sockets, _ = transport
    sockets["left"] = FakeSocket((100, RuntimeError("unavailable"), 300))
    output = tmp_path / "watch.jsonl"

    summary = watch.watch_grippers(config, duration_s=0.21, fps=10, output=output)

    assert summary["preflight_ok"] is True
    assert summary["sides"]["left"] == {
        "attempts": 3,
        "sample_count": 2,
        "min": 0.1,
        "max": 0.3,
        "changed": True,
        "errors": 1,
    }
    assert summary["sides"]["right"]["changed"] is False
    assert summary["sides"]["right"]["sample_count"] == 3
    records = [json.loads(line) for line in output.read_text().splitlines()]
    samples = [record for record in records if record["event"] == "sample" and record["side"] == "left"]
    assert [sample["normalized"] for sample in samples] == [0.1, None, 0.3]
    assert samples[1]["raw_position"] is samples[1]["raw_state"] is None
    assert "unavailable" in samples[1]["error"]
    assert samples[0]["raw_state"]["state_info"]["pos"] == [100]
    assert records[-1] == summary
    assert "ERROR" in capsys.readouterr().out
    assert all(sock.closed for sock in sockets.values())
    assert all(
        payload["command"] in watch.READ_COMMANDS for sock in sockets.values() for payload in sock.sent
    )


def test_rm_plus_enabled_but_device_offline_stops_without_zero(config, transport, clock, tmp_path, capsys):
    sockets, _ = transport
    for side in sockets:
        sockets[side] = FakeSocket(base={"get_state": False})
    output = tmp_path / "offline.jsonl"

    summary = watch.watch_grippers(config, duration_s=1, output=output)

    assert summary["preflight_ok"] is False
    records = [json.loads(line) for line in output.read_text().splitlines()]
    preflights = [record for record in records if record["event"] == "preflight"]
    assert len(preflights) == 2
    assert all(record["mode_enabled"] is True and record["device_online"] is False for record in preflights)
    assert all(
        record["base_response"]["get_state"] is False and "get_state=false" in record["error"]
        for record in preflights
    )
    assert not any(record["event"] == "sample" for record in records)
    assert all(
        item["sample_count"] == 0 and item["min"] is item["max"] is None and item["errors"] == 1
        for item in summary["sides"].values()
    )
    text = capsys.readouterr().out
    assert "enabled does not mean device online" in text
    assert "normalized=" not in text
    assert all(sock.closed for sock in sockets.values())
    assert all(
        [payload["command"] for payload in sock.sent] == ["get_rm_plus_mode", "get_rm_plus_base_info"]
        for sock in sockets.values()
    )


def test_disabled_rm_plus_is_not_enabled_or_sampled(config, transport, clock):
    sockets, _ = transport
    sockets["left"] = FakeSocket(mode=0)

    summary = watch.watch_grippers(config, duration_s=0.1)

    assert summary["preflight_ok"] is False
    assert sockets["left"].sent == [{"command": "get_rm_plus_mode"}]
    assert all(item["attempts"] == 0 for item in summary["sides"].values())
    assert all(sock.closed for sock in sockets.values())


@pytest.mark.parametrize(
    "response",
    [
        {"get_state": False},
        {"state_info": {"sys_err": 1, "dof_err": [0], "pos": [0]}},
        {"state_info": {"sys_err": 0, "dof_err": [0]}},
    ],
)
def test_invalid_state_preserves_raw_error_and_never_returns_zero(
    config, transport, clock, tmp_path, response
):
    sockets, _ = transport
    sockets["left"] = FakeSocket((response,))
    output = tmp_path / "invalid-state.jsonl"

    summary = watch.watch_grippers(config, duration_s=0.1, output=output)

    samples = [
        record
        for line in output.read_text().splitlines()
        if (record := json.loads(line))["event"] == "sample" and record["side"] == "left"
    ]
    assert len(samples) == 1
    assert samples[0]["normalized"] is samples[0]["raw_position"] is None
    assert samples[0]["raw_state"] == {"command": "get_rm_plus_state_info", **response}
    assert "error" in samples[0]
    assert summary["sides"]["left"]["sample_count"] == 0
    assert summary["sides"]["left"]["errors"] == 1
    assert summary["sides"]["left"]["min"] is summary["sides"]["left"]["max"] is None
    assert all(sock.closed for sock in sockets.values())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_feedback_is_logged_as_error_and_monitor_continues(
    config, transport, clock, tmp_path, value
):
    sockets, _ = transport
    sockets["left"] = FakeSocket((value, 425))
    output = tmp_path / "nonfinite.jsonl"

    summary = watch.watch_grippers(config, duration_s=0.11, fps=10, output=output)

    def reject_nonfinite_constant(token):
        pytest.fail(f"Invalid JSON constant in diagnostic log: {token}")

    records = [
        json.loads(line, parse_constant=reject_nonfinite_constant) for line in output.read_text().splitlines()
    ]
    samples = [item for item in records if item["event"] == "sample" and item["side"] == "left"]
    assert len(samples) == 2
    assert samples[0]["normalized"] is samples[0]["raw_position"] is None
    assert samples[0]["raw_state"]["state_info"]["pos"] == [str(value)]
    assert "finite number" in samples[0]["error"]
    assert samples[1]["normalized"] == 0.425
    assert summary["sides"]["left"]["errors"] == 1
    assert summary["sides"]["left"]["sample_count"] == 1
    assert records[-1] == summary
    assert all(sock.closed for sock in sockets.values())


@pytest.mark.parametrize(
    "base", [{"dof": 2, "pos_low": [0], "pos_up": [1000]}, {"dof": 1, "pos_low": [0], "pos_up": [100]}]
)
def test_metadata_must_match_single_gripper_and_scale(config, transport, clock, base):
    sockets, _ = transport
    sockets["left"] = FakeSocket(base=base)

    summary = watch.watch_grippers(config, duration_s=0.1)

    assert summary["preflight_ok"] is False
    assert all(item["attempts"] == 0 for item in summary["sides"].values())
    assert all(sock.closed for sock in sockets.values())


def test_both_grippers_are_sampled_concurrently(config, transport, clock):
    sockets, _ = transport
    barrier = Barrier(2)
    sockets.update({side: FakeSocket(barrier=barrier) for side in sockets})

    summary = watch.watch_grippers(config, duration_s=0.1, fps=10)

    assert all(item["sample_count"] == 1 and item["errors"] == 0 for item in summary["sides"].values())


def test_ctrl_c_closes_sockets_executor_and_writes_summary(config, transport, clock, monkeypatch, tmp_path):
    sockets, _ = transport
    executors = []
    real_executor = watch.ThreadPoolExecutor

    def executor_factory(**kwargs):
        executor = real_executor(**kwargs)
        executors.append(executor)
        return executor

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(watch, "ThreadPoolExecutor", executor_factory)
    monkeypatch.setattr(watch.time, "sleep", interrupt)
    output = tmp_path / "interrupt.jsonl"

    summary = watch.watch_grippers(config, duration_s=30, output=output)

    assert summary["interrupted"] is True
    assert all(sock.closed for sock in sockets.values())
    assert all(not thread.is_alive() for executor in executors for thread in executor._threads)
    assert json.loads(output.read_text().splitlines()[-1]) == summary


def test_existing_jsonl_is_never_overwritten_or_connected(config, transport, tmp_path):
    _, attempts = transport
    output = tmp_path / "existing.jsonl"
    output.write_text("preserve this\n")

    with pytest.raises(FileExistsError):
        watch.watch_grippers(config, duration_s=0.1, output=output)

    assert output.read_text() == "preserve this\n"
    assert attempts == []


@pytest.mark.parametrize("duration_s,fps", [(0, 10), (float("inf"), 10), (1, 0), (1, 31), (1, float("nan"))])
def test_invalid_timing_rejected_without_connection(config, transport, duration_s, fps):
    _, attempts = transport
    with pytest.raises(ValueError, match="duration-s"):
        watch.watch_grippers(config, duration_s=duration_s, fps=fps)
    assert attempts == []

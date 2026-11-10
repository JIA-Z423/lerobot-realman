# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""No device connections: all TCP and serial transports are simulated."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/realman/check_realman_readonly.py"
spec = importlib.util.spec_from_file_location("realman_readonly", SCRIPT)
readonly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readonly)

# Response structures captured from the two connected RM+ grippers on 2026-11-01.
# Keep these in the test rather than depending on a temporary hardware evidence file.
RM_PLUS_BASE_INFO = {
    "angle_low": [0],
    "angle_up": [0],
    "bee": 0,
    "bv": "V0.1",
    "check": 0,
    "dof": 1,
    "force": True,
    "force_low": [0],
    "force_up": [100],
    "hand": 0,
    "hv": "V0.1",
    "id": 1,
    "manu": "ZX",
    "pos_low": [0],
    "pos_up": [1000],
    "speed_low": [0],
    "speed_up": [5000],
    "sv": "V1.154",
    "touch": False,
    "type": 1,
}
RM_PLUS_STATE_INFO = {
    "angle": [0],
    "current": [1],
    "dof_err": [0],
    "dof_state": [2],
    "force": [0],
    "pos": [0],
    "speed": [0],
    "sys_err": 0,
    "sys_state": 0,
}


class FakeSocket:
    def __init__(self, rm_plus_mode=115200, response_overrides=None):
        self.sent = []
        self.pending = b""
        self.closed = False
        self.rm_plus_mode = rm_plus_mode
        self.response_overrides = response_overrides or {}

    def settimeout(self, timeout):
        pass

    def sendall(self, data):
        payload = json.loads(data)
        self.sent.append(payload)
        command = payload["command"]
        response = {"command": command}
        if command == "get_gripper_state":
            response.update(actpos=321, status=1)
        elif command == "get_joint_degree":
            response["joint"] = [0] * 6
        elif command in {"get_tool_RS485_mode", "get_controller_RS485_mode"}:
            response.update(mode=0, baudrate=115200, timeout=1)
        elif command == "get_rm_plus_mode":
            response["mode"] = self.rm_plus_mode
        elif command == "get_rm_plus_base_info":
            response["base_info"] = RM_PLUS_BASE_INFO
        elif command == "get_rm_plus_state_info":
            response["state_info"] = RM_PLUS_STATE_INFO
        elif command == "read_multiple_holding_registers":
            assert 2 < payload["num"] < 13
            response["data"] = [0, 0, 1, 65, 0x12, 0x34]
        elif command == "read_holding_registers":
            response["data"] = 1
        response = self.response_overrides.get(command, response)
        # A plausible asynchronous notification must not satisfy the request.
        self.pending += (json.dumps({"command": "get_wrong_status", "actpos": 999}) + "\r\n").encode()
        self.pending += (json.dumps(response) + "\r\n").encode()

    def recv(self, size):
        chunk, self.pending = self.pending[:size], self.pending[size:]
        return chunk

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "command",
    [
        "movej_canfd",
        "set_gripper_position",
        "set_gripper_route",
        "set_modbus_mode",
        "set_tool_voltage",
        "set_tool_RS485",
        "set_controller_RS485",
        "set_rm_plus_mode",
        "set_rm_plus_position",
    ],
)
def test_transport_refuses_writes_before_sending(monkeypatch, command):
    sock = FakeSocket()
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *args, **kwargs: sock)
    client = readonly.ReadOnlyJsonClient("offline", 8080, 1)
    with pytest.raises(ValueError, match="Refusing non-read-only"):
        client.request({"command": command})
    assert sock.sent == []
    client.close()


def test_follower_uses_only_status_queries_and_correlates_commands(monkeypatch):
    from lerobot.common.realman.client import RealManSocketClient

    def forbidden(*args, **kwargs):
        pytest.fail("The read-only check must not initialize or command a robot")

    monkeypatch.setattr(RealManSocketClient, "configure", forbidden)
    monkeypatch.setattr(RealManSocketClient, "send_joint_positions", forbidden)
    sock = FakeSocket()
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *args, **kwargs: sock)
    result = readonly.read_follower({"host": "offline"}, samples=2, timeout_s=1)
    assert sock.closed
    assert len(sock.sent) == 12
    assert sum(payload["command"] == "get_tool_voltage" for payload in sock.sent) == 1
    assert {payload["command"] for payload in sock.sent} <= readonly.READ_COMMANDS
    assert all("error" not in query for query in result["queries"])
    assert all(query["response"]["command"] == query["payload"]["command"] for query in result["queries"])
    grippers = [query for query in result["queries"] if query["payload"]["command"] == "get_gripper_state"]
    assert [query["response"]["actpos"] for query in grippers] == [321, 321]
    assert result["native_gripper"]["samples"] == grippers
    assert all(query["source"] == "native_gripper" for query in grippers)
    assert result["rm_plus"]["status"] == "enabled"
    assert result["rm_plus"]["enabled"] is True
    assert result["rm_plus"]["mode"] == result["rm_plus"]["baudrate"] == 115200
    assert result["rm_plus"]["base_info"] == RM_PLUS_BASE_INFO
    assert [query["response"]["state_info"] for query in result["rm_plus"]["samples"]] == [
        RM_PLUS_STATE_INFO,
        RM_PLUS_STATE_INFO,
    ]
    assert result["rm_plus"]["errors"] == []
    assert result["modbus_probe"]["enabled"] is False
    assert result["modbus_probe"]["queries"] == []
    assert not any(payload["command"].startswith("read_") for payload in sock.sent)
    for command in (
        "get_tool_RS485_mode",
        "get_controller_RS485_mode",
        "get_rm_plus_mode",
        "get_rm_plus_base_info",
    ):
        assert sum(payload["command"] == command for payload in sock.sent) == 1
    sampled_queries = [query["payload"]["command"] for query in result["queries"] if query["sample"] >= 0]
    assert sampled_queries == ["get_joint_degree", "get_gripper_state", "get_rm_plus_state_info"] * 2


def test_rm_plus_disabled_is_reported_without_reading_state_or_enabling(monkeypatch):
    sock = FakeSocket(rm_plus_mode=0)
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *args, **kwargs: sock)

    result = readonly.read_follower({"host": "offline"}, samples=2, timeout_s=1)

    assert result["rm_plus"]["status"] == "disabled"
    assert result["rm_plus"]["enabled"] is False
    assert result["rm_plus"]["mode"] == 0
    assert result["rm_plus"]["samples"] == []
    assert result["rm_plus"]["errors"] == []
    commands = [payload["command"] for payload in sock.sent]
    assert commands == [
        "get_arm_software_info",
        "get_tool_voltage",
        "get_tool_RS485_mode",
        "get_controller_RS485_mode",
        "get_rm_plus_mode",
        "get_joint_degree",
        "get_gripper_state",
        "get_joint_degree",
        "get_gripper_state",
    ]
    assert sock.closed


@pytest.mark.parametrize("mode", [9600, 115200, 256000, 460800])
def test_documented_rm_plus_modes_read_base_and_state_without_writes(monkeypatch, mode):
    sock = FakeSocket(rm_plus_mode=mode)
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *args, **kwargs: sock)

    result = readonly.read_follower({"host": "offline"}, samples=1, timeout_s=1)

    assert result["rm_plus"]["enabled"] is True
    assert result["rm_plus"]["baudrate"] == mode
    assert result["rm_plus"]["samples"][0]["response"]["state_info"] == RM_PLUS_STATE_INFO
    assert all(payload["command"].startswith("get_") for payload in sock.sent)


@pytest.mark.parametrize("command", ["get_rm_plus_mode", "get_rm_plus_base_info", "get_rm_plus_state_info"])
def test_rm_plus_get_state_false_is_an_explicit_diagnostic_error(monkeypatch, command):
    response = {"command": command, "get_state": False}
    sock = FakeSocket(response_overrides={command: response})
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *args, **kwargs: sock)

    result = readonly.read_follower({"host": "offline"}, samples=1, timeout_s=1)

    assert result["rm_plus"]["status"] == "error"
    assert any(
        error["command"] == command and "get_state=false" in error["error"]
        for error in result["rm_plus"]["errors"]
    )
    query = next(query for query in result["queries"] if query["payload"]["command"] == command)
    assert query["response"] == response
    assert query["error"] == "Controller returned get_state=false"
    assert all(payload["command"].startswith("get_") for payload in sock.sent)
    if command == "get_rm_plus_mode":
        assert not any(
            payload["command"] in {"get_rm_plus_base_info", "get_rm_plus_state_info"} for payload in sock.sent
        )


def test_direct_modbus_queries_require_explicit_opt_in(monkeypatch):
    sock = FakeSocket()
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *args, **kwargs: sock)

    result = readonly.read_follower({"host": "offline"}, samples=2, timeout_s=1, probe_modbus=True)

    assert len(sock.sent) == 20
    assert result["modbus_probe"]["enabled"] is True
    assert len(result["modbus_probe"]["queries"]) == 8
    registers = [query for query in result["modbus_probe"]["queries"] if "raw_u32_big_endian" in query]
    assert [query["raw_u32_big_endian"] for query in registers] == [321] * 4
    assert {payload["command"] for payload in sock.sent} <= readonly.READ_COMMANDS
    assert all("error" not in query for query in result["queries"])


def test_failed_connection_is_not_retried(monkeypatch):
    attempts = []

    def failed(*args, **kwargs):
        attempts.append(args)
        raise TimeoutError("simulated")

    monkeypatch.setattr(readonly.socket, "create_connection", failed)
    result = readonly.read_follower({"host": "offline"}, samples=3, timeout_s=1)
    assert len(attempts) == 1
    assert result["connection_error"] == "TimeoutError: simulated"
    assert result["queries"] == []


@pytest.mark.parametrize(
    "command,response",
    [
        ("get_arm_software_info", {"command": "arm_software_info", "version": "test"}),
        ("get_joint_degree", {"state": "joint_degree", "joint": [0] * 6}),
        ("get_tool_voltage", {"state": "tool_voltage_state", "voltage_type": 3}),
    ],
)
def test_known_rm65_response_aliases(monkeypatch, command, response):
    sock = FakeSocket()
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *args, **kwargs: sock)

    def sendall(data):
        sock.sent.append(json.loads(data))
        sock.pending = (json.dumps(response) + "\r\n").encode()

    sock.sendall = sendall
    client = readonly.ReadOnlyJsonClient("offline", 8080, 1)
    query = client.request({"command": command})
    assert query["response"] == response
    assert "error" not in query
    assert not readonly.matches_response("get_gripper_state", response)
    client.close()


def test_master_rejects_custom_query_without_opening_port(monkeypatch):
    from lerobot.teleoperators.rm_aloha_master import RMAlohaMaster

    def forbidden(*args, **kwargs):
        pytest.fail("Custom query must be rejected before constructing the serial device")

    monkeypatch.setattr(RMAlohaMaster, "__init__", forbidden)
    result = readonly.read_master({"port": "/offline-master", "init_hex_command": "FFFF"}, samples=1)
    assert "Refusing custom master" in result["error"]


def test_master_uses_only_fixed_query_and_decoder(monkeypatch):
    from lerobot.teleoperators.rm_aloha_master import RMAlohaMaster, rm_aloha_master as master_module

    class FakeSerial:
        def __init__(self, *args, **kwargs):
            self.port = None
            self.is_open = False
            self.sent = []

        def open(self):
            self.is_open = True

        def write(self, data):
            self.sent.append(data)

        def close(self):
            self.is_open = False

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only master check must not call connect/configure")

    instances = []

    def serial_factory(*args, **kwargs):
        device = FakeSerial()
        instances.append(device)
        return device

    encoded = list("AA55" + "00" * 39)
    for (start, end), value in zip(RMAlohaMaster._field_slices, [10000] * 6 + [321], strict=True):
        encoded[start:end] = value.to_bytes(4, "little", signed=True).hex()
    monkeypatch.setattr(master_module.serial, "Serial", serial_factory)
    monkeypatch.setattr(RMAlohaMaster, "connect", forbidden)
    monkeypatch.setattr(RMAlohaMaster, "configure", forbidden)
    monkeypatch.setattr(RMAlohaMaster, "_read_encoded_frame", lambda self: "".join(encoded))
    result = readonly.read_master({"port": "/offline-master", "frame_header": "AA55"}, samples=2)
    assert "error" not in result
    assert instances[0].sent == [readonly.MASTER_QUERY] * 2
    assert not instances[0].is_open
    assert [sample["state"]["gripper.pos"] for sample in result["samples"]] == [0.321, 0.321]


@pytest.fixture
def summary_inputs():
    config = {"robot": {}}
    report = {"followers": {}, "masters": {}, "cameras": {"configured": {}}, "skipped": []}
    for side in ("left", "right"):
        config["robot"][f"{side}_arm_config"] = {
            "has_gripper": True,
            "gripper_feedback_source": "rm_plus",
            "gripper_scale": 1000.0,
        }
        responses = {
            "get_joint_degree": {"joint": [0] * 6},
            "get_gripper_state": {"status": 0, "enable": 0, "actpos": 0},
            "get_rm_plus_mode": {"mode": 115200},
            "get_rm_plus_base_info": {"base_info": deepcopy(RM_PLUS_BASE_INFO)},
            "get_rm_plus_state_info": {"state_info": deepcopy(RM_PLUS_STATE_INFO)},
        }
        report["followers"][side] = {
            "queries": [
                {"payload": {"command": command}, "response": response}
                for command, response in responses.items()
            ]
        }
        report["masters"][side] = {
            "exists": True,
            "port": f"/offline-{side}",
            "samples": [{"state": {**{f"joint_{i}.pos": 0.0 for i in range(1, 7)}, "gripper.pos": 0.0}}],
        }
        report["cameras"]["configured"][f"{side}_wrist"] = {
            "config": {"width": 640, "height": 480},
            "frame": {"width": 640, "height": 480},
        }
    return report, config


def test_summary_accepts_closed_grippers_and_ignores_mismatched_native_feedback(summary_inputs, capsys):
    report, config = summary_inputs
    # RM+ confirms the right gripper is open while the old native interface still says zero.
    report["followers"]["right"]["queries"][-1]["response"]["state_info"]["pos"] = [300]
    assert readonly.print_summary(report, config)
    output = capsys.readouterr().out
    assert "通过 8 项，失败 0 项，跳过 0 项" in output
    assert "位置=0.000" in output
    assert "位置=0.300" in output


@pytest.mark.parametrize(
    "patch",
    [
        {"sys_err": 1},
        {"sys_error": 1},
        {"dof_err": [1]},
        {"pos": []},
        {"pos": [True]},
        {"pos": [float("nan")]},
        {"pos": [1001]},
    ],
)
def test_summary_rejects_invalid_rm_plus_feedback(summary_inputs, capsys, patch):
    report, config = summary_inputs
    report["followers"]["left"]["queries"][-1]["response"]["state_info"].update(patch)
    assert not readonly.print_summary(report, config)
    assert "[失败] 左夹爪" in capsys.readouterr().out


def test_summary_does_not_treat_enabled_mode_as_online_device(summary_inputs, capsys):
    report, config = summary_inputs
    report["followers"]["left"]["queries"][-2] = {
        "payload": {"command": "get_rm_plus_base_info"},
        "response": {"get_state": False},
        "error": "Controller returned get_state=false",
    }
    assert not readonly.print_summary(report, config)
    assert "get_state=false" in capsys.readouterr().out


def test_summary_checks_master_samples_and_camera_stop_errors(summary_inputs, capsys):
    report, config = summary_inputs
    report["masters"]["left"]["samples"][0]["state"]["gripper.pos"] = float("inf")
    report["cameras"]["configured"]["right_wrist"]["stop_error"] = "USB disconnected"
    assert not readonly.print_summary(report, config)
    output = capsys.readouterr().out
    assert "[失败] 左主手" in output
    assert "[失败] 右腕相机：USB disconnected" in output


def test_summary_does_not_pass_when_everything_is_skipped(summary_inputs, capsys):
    report, config = summary_inputs
    report["skipped"] = ["followers", "masters", "cameras"]
    assert not readonly.print_summary(report, config)
    assert "通过 0 项，失败 0 项，跳过 8 项" in capsys.readouterr().out


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("mode", ["summary", "summary-with-json", "json-only"])
def test_summary_cli_uses_default_config_without_requiring_json(
    summary_inputs, monkeypatch, tmp_path, failed, mode
):
    report, config = summary_inputs
    # All hardware entry points are replaced; any accidental real TCP access fails the test.
    monkeypatch.setattr(readonly.socket, "create_connection", lambda *a, **k: pytest.fail("Real TCP access"))
    monkeypatch.setattr(readonly, "serial_inventory", lambda: {})
    monkeypatch.setattr(readonly.yaml, "safe_load", lambda text: config)
    monkeypatch.setattr(readonly.Path, "read_text", lambda self: "offline config")
    monkeypatch.setattr(readonly, "read_follower", lambda *a, **k: report["followers"]["left"])
    monkeypatch.setattr(readonly, "read_master", lambda *a, **k: report["masters"]["left"])
    monkeypatch.setattr(readonly, "read_cameras", lambda *a, **k: report["cameras"])
    config["teleop"] = {"left_arm_config": {}, "right_arm_config": {}}
    monkeypatch.chdir(tmp_path)
    argv = [str(SCRIPT)]
    if mode != "json-only":
        argv.append("--summary")
    if mode != "summary":
        argv.extend(["--output", "report.json"])
    monkeypatch.setattr("sys.argv", argv)
    if failed:
        report["followers"]["left"]["queries"][-1]["response"]["state_info"]["sys_err"] = 1
    if failed and mode != "json-only":
        with pytest.raises(SystemExit) as exc:
            readonly.main()
        assert exc.value.code == 1
    else:
        readonly.main()
    if mode == "summary":
        assert not list(tmp_path.iterdir())
    else:
        with (tmp_path / "report.json").open() as stream:
            saved = json.load(stream)
        assert saved["followers"]["left"] == report["followers"]["left"]

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from lerobot.common.realman.client import (
    BaseRealManClient,
    RealManAlohaSDKClient,
    RealManClientConfig,
    RealManSDKClient,
    RealManSocketClient,
)
from lerobot.teleoperators.rm_aloha_master.config_rm_aloha_master import RMAlohaMasterConfig
from lerobot.teleoperators.rm_aloha_master.rm_aloha_master import RMAlohaMaster


@pytest.mark.parametrize("source", ["unknown", "RM+", ""])
def test_client_config_rejects_unknown_gripper_feedback_source(source):
    with pytest.raises(ValueError, match="gripper_feedback_source must be"):
        RealManClientConfig(host="offline", gripper_feedback_source=source)


@pytest.mark.parametrize("backend", ["vendor_sdk", "rm_aloha_sdk"])
def test_client_config_rejects_rm_plus_for_unsupported_sdk(backend):
    with pytest.raises(ValueError, match="rm_plus gripper feedback is only supported"):
        RealManClientConfig(host="offline", sdk_backend=backend, gripper_feedback_source="rm_plus")


def test_client_config_rejects_rm_plus_with_modbus_register():
    with pytest.raises(ValueError, match="cannot be combined with gripper_position_register"):
        RealManClientConfig(
            host="offline", gripper_feedback_source="rm_plus", gripper_position_register=0x0609
        )


def test_rm_plus_accepts_actual_single_dof_feedback_with_zero_position():
    client = RealManSocketClient(
        RealManClientConfig(host="offline", gripper_feedback_source="rm_plus", initialize_gripper_route=False)
    )
    requests = []
    client._send_payload = requests.append
    # RM65 V1.7.4 with CTAG2F120 RM_ARM+ firmware: the state values are
    # preserved without guessing undocumented meanings for sys_state/dof_state.
    client._read_response = lambda: {
        "command": "get_rm_plus_state_info",
        "state_info": {
            "angle": [0],
            "current": [1],
            "dof_err": [0],
            "dof_state": [2],
            "force": [0],
            "pos": [0],
            "speed": [0],
            "sys_err": 0,
            "sys_state": 0,
        },
    }

    assert client._read_gripper_value() == 0.0
    assert requests == [{"command": "get_rm_plus_state_info"}]


@pytest.mark.parametrize("error_key", ["sys_err", "sys_error"])
def test_rm_plus_accepts_finite_nonzero_position(error_key):
    client = RealManSocketClient(RealManClientConfig(host="offline", gripper_feedback_source="rm_plus"))
    client._send_command_with_expected_key = lambda payload, expected_key: {
        "state_info": {"pos": [425], "dof_err": [0], error_key: 0}
    }

    assert client._read_gripper_value() == 0.425


@pytest.mark.parametrize(
    "state, message",
    [
        (None, "Expected RM\\+ state_info object"),
        ({"pos": [0], "dof_err": [0], "sys_err": 1}, "sys_err=1"),
        ({"pos": [0], "dof_err": [0], "sys_error": 2}, "sys_error=2"),
        ({"pos": [0], "dof_err": [0], "sys_state": 1, "sys_err": 0}, "sys_state=1"),
        ({"pos": [0], "dof_err": [0]}, "Missing RM\\+ system error field"),
        ({"pos": [0], "dof_err": [5], "sys_err": 0}, "one error-free"),
        ({"pos": [0], "dof_err": [], "sys_err": 0}, "one error-free"),
        ({"pos": [0], "dof_err": [0, 0], "sys_err": 0}, "one error-free"),
        ({"pos": [0], "sys_err": 0}, "one error-free"),
        ({"dof_err": [0], "sys_err": 0}, "one RM\\+ gripper position"),
        ({"pos": [], "dof_err": [0], "sys_err": 0}, "one RM\\+ gripper position"),
        ({"pos": [0, 1], "dof_err": [0], "sys_err": 0}, "one RM\\+ gripper position"),
        ({"pos": [True], "dof_err": [0], "sys_err": 0}, "finite number"),
        ({"pos": [float("nan")], "dof_err": [0], "sys_err": 0}, "finite number"),
        ({"pos": [float("inf")], "dof_err": [0], "sys_err": 0}, "finite number"),
    ],
)
def test_rm_plus_rejects_invalid_feedback_without_fallback(state, message):
    client = RealManSocketClient(RealManClientConfig(host="offline", gripper_feedback_source="rm_plus"))
    client._last_gripper_value = 0.5
    requests = []

    def response(payload, expected_key):
        requests.append((payload, expected_key))
        return {"state_info": state}

    client._send_command_with_expected_key = response
    with pytest.raises(RuntimeError, match=message):
        client._read_gripper_value()
    assert requests == [({"command": "get_rm_plus_state_info"}, "state_info")]


def test_rm_plus_reports_get_state_false_immediately():
    client = RealManSocketClient(RealManClientConfig(host="offline", gripper_feedback_source="rm_plus"))
    client._send_payload = lambda payload: None
    client._read_response = lambda: {"command": "get_rm_plus_state_info", "get_state": False}

    with pytest.raises(RuntimeError, match="get_state=false"):
        client._read_gripper_value()


@pytest.mark.parametrize("positions", [{"joint_1": 1.0}, {"gripper": 0.5}, {"joint_1": 1.0, "gripper": 0.5}])
def test_rm_plus_rejects_all_targets_before_sending_any_commands(positions):
    client = RealManSocketClient(RealManClientConfig(host="offline", gripper_feedback_source="rm_plus"))
    requests = []
    client._send_payload = requests.append
    client._send_command = requests.append

    with pytest.raises(RuntimeError, match="RM\\+ feedback support is read-only"):
        client.send_joint_positions(positions)
    assert requests == []


def test_rm_plus_configure_never_sets_device_mode_or_route():
    client = RealManSocketClient(
        RealManClientConfig(host="offline", gripper_feedback_source="rm_plus", initialize_gripper_route=False)
    )
    requests = []
    client._send_command = requests.append

    client.configure()
    assert requests == []
    client.config.initialize_gripper_route = True
    with pytest.raises(RuntimeError, match="no initialization commands were sent"):
        client.configure()
    assert requests == []


def test_rm_plus_follower_preserves_nonzero_feedback_in_dataset(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("datasets")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.realman_follower import RealManFollower, RealManFollowerConfig
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    robot = RealManFollower(
        RealManFollowerConfig(
            host="offline",
            calibration_dir=tmp_path / "calibration",
            gripper_feedback_source="rm_plus",
            initialize_gripper_route=False,
        )
    )
    assert robot.client.config.gripper_feedback_source == "rm_plus"
    robot.client._is_connected = True
    requests = []

    def response(payload, expected_key):
        requests.append(payload)
        if payload["command"] == "get_joint_degree":
            return {"joint": [1000, 2000, 3000, 4000, 5000, 6000]}
        assert payload == {"command": "get_rm_plus_state_info"}
        return {"state_info": {"pos": [425], "dof_err": [0], "sys_err": 0}}

    robot.client._send_command_with_expected_key = response
    features = hw_to_dataset_features(robot.observation_features, prefix="observation", use_video=False)
    dataset = LeRobotDataset.create(
        repo_id="local/rm_plus_offline",
        fps=30,
        root=tmp_path / "dataset",
        features=features,
        use_videos=False,
    )
    try:
        obs = robot.get_observation()
        assert obs["gripper.pos"] == 0.425
        for _ in range(3):
            dataset.add_frame(
                {**build_dataset_frame(dataset.features, obs, prefix="observation"), "task": "offline"}
            )
        dataset.save_episode()
    finally:
        dataset.finalize()
        robot.client._is_connected = False
    table = pq.read_table(next((tmp_path / "dataset" / "data").rglob("*.parquet")))
    index = dataset.features["observation.state"]["names"].index("gripper.pos")
    assert [row[index] for row in table["observation.state"].to_pylist()] == pytest.approx([0.425] * 3)
    assert requests == [{"command": "get_joint_degree"}, {"command": "get_rm_plus_state_info"}]


def test_socket_client_scales_gripper_state():
    client = RealManSocketClient(
        RealManClientConfig(host="127.0.0.1", has_gripper=True, gripper_scale=1000.0)
    )
    client._send_command_with_expected_key = lambda payload, expected_key: {"actpos": 425}

    assert client._read_gripper_value() == 0.425


def test_socket_client_uses_configured_gripper_state_field():
    client = RealManSocketClient(
        RealManClientConfig(
            host="127.0.0.1",
            has_gripper=True,
            gripper_scale=1000.0,
            gripper_state_position_key="position",
        )
    )
    client._send_command_with_expected_key = lambda payload, expected_key: {
        "actpos": 0,
        "position": 425,
    }

    assert client._read_gripper_value() == 0.425


def test_socket_client_reads_gripper_position_from_feedback_registers():
    client = RealManSocketClient(
        RealManClientConfig(
            host="127.0.0.1",
            gripper_scale=1000.0,
            gripper_position_register=0x0609,
            gripper_modbus_port=1,
            gripper_modbus_device=1,
        )
    )
    requests = []

    def read_registers(payload, expected_key):
        requests.append((payload, expected_key))
        return {"data": [0x00, 0x01, 0x86, 0xA0, 0x12, 0x34]}

    client._send_command_with_expected_key = read_registers

    assert client._read_gripper_value() == 100.0
    assert requests == [
        (
            {
                "command": "read_multiple_holding_registers",
                "port": 1,
                "address": 0x0609,
                "num": 3,
                "device": 1,
            },
            "data",
        )
    ]


@pytest.mark.parametrize("scale_name", ["joint_scale", "gripper_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_client_config_rejects_invalid_scales(scale_name, value):
    with pytest.raises(ValueError, match=f"{scale_name} must be a positive finite number"):
        RealManClientConfig(host="127.0.0.1", **{scale_name: value})


@pytest.mark.parametrize("address", [-1, 65534, 65535, 1.5, True])
def test_client_config_rejects_invalid_gripper_register_address(address):
    with pytest.raises(ValueError, match="gripper_position_register must be an integer"):
        RealManClientConfig(host="127.0.0.1", gripper_position_register=address)


def test_register_feedback_can_read_batch_at_highest_valid_address():
    client = RealManSocketClient(RealManClientConfig(host="offline", gripper_position_register=65533))
    requests = []

    def read_registers(payload, expected_key):
        requests.append(payload)
        return {"data": [0, 0, 1, 169, 255, 255]}

    client._send_command_with_expected_key = read_registers

    assert client._read_gripper_value() == 0.425
    assert requests[0]["address"] + requests[0]["num"] - 1 == 65535


@pytest.mark.parametrize("backend", ["vendor_sdk", "rm_aloha_sdk"])
def test_client_config_rejects_register_feedback_for_unsupported_sdk(backend):
    with pytest.raises(ValueError, match="only supported by the socket_json backend"):
        RealManClientConfig(host="127.0.0.1", sdk_backend=backend, gripper_position_register=0x0609)


@pytest.mark.parametrize("state_type", [dict, SimpleNamespace])
@pytest.mark.parametrize("state", [{"status": 0}, {"status": 2}, {"error": 5}])
def test_gripper_state_rejects_invalid_feedback_status(state_type, state):
    client = BaseRealManClient(RealManClientConfig(host="127.0.0.1"))
    payload = state_type(actpos=0, **state)

    with pytest.raises(RuntimeError, match="Invalid gripper feedback"):
        client._get_gripper_state_position(payload, "get_gripper_state")


@pytest.mark.parametrize("state_type", [dict, SimpleNamespace])
def test_gripper_state_accepts_valid_feedback_even_when_not_enabled(state_type):
    client = BaseRealManClient(RealManClientConfig(host="127.0.0.1"))
    payload = state_type(actpos=425, status=1, error=0, enable=0, enable_state=False)

    assert client._get_gripper_state_position(payload, "get_gripper_state") == 0.425


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_gripper_state_rejects_invalid_position(value):
    client = BaseRealManClient(RealManClientConfig(host="127.0.0.1"))

    with pytest.raises(RuntimeError, match="Gripper position must be a finite number"):
        client._get_gripper_state_position({"actpos": value}, "get_gripper_state")


@pytest.mark.parametrize("state_type", [dict, SimpleNamespace])
def test_gripper_state_reports_missing_configured_position(state_type):
    client = BaseRealManClient(RealManClientConfig(host="127.0.0.1", gripper_state_position_key="position"))

    with pytest.raises(RuntimeError, match="Gripper state field 'position' was not found"):
        client._get_gripper_state_position(state_type(actpos=425), "get_gripper_state")


@pytest.mark.parametrize(
    "data",
    [
        0,
        [0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, 0, 256, 0, 0, 0],
        [0, 0, 1.5, 0, 0, 0],
        [0, 0, True, 0, 0, 0],
        [0, 0, 0, 0, 256, 0],
        [0, 0, 0, 0, 0, True],
    ],
)
def test_register_feedback_rejects_malformed_data_without_using_cached_position(data):
    client = RealManSocketClient(RealManClientConfig(host="127.0.0.1", gripper_position_register=0x0609))
    client._last_gripper_value = 0.5
    client._send_command_with_expected_key = lambda payload, expected_key: {"data": data}

    with pytest.raises(RuntimeError, match="Expected six big-endian register bytes"):
        client._read_gripper_value()


def test_register_feedback_error_does_not_fall_back_to_gripper_state():
    client = RealManSocketClient(RealManClientConfig(host="127.0.0.1", gripper_position_register=0x0609))
    requests = []
    client._send_payload = requests.append
    client._read_response = lambda: {"command": "read_multiple_holding_registers", "read_state": False}

    with pytest.raises(RuntimeError, match="read_state=false"):
        client._read_gripper_value()

    assert len(requests) == 1
    assert requests[0]["command"] == "read_multiple_holding_registers"


def test_expected_response_ignores_other_commands_and_sends_request_once():
    client = RealManSocketClient(RealManClientConfig(host="127.0.0.1"))
    requests = []
    client._send_payload = requests.append
    responses = iter(
        [
            {"command": "another_command", "actpos": 0, "read_state": False},
            {"command": "get_gripper_state", "actpos": 425},
        ]
    )
    client._read_response = lambda: next(responses)

    assert client._read_gripper_value() == 0.425
    assert requests == [{"command": "get_gripper_state"}]


def test_expected_response_accepts_legacy_response_without_command():
    client = RealManSocketClient(RealManClientConfig(host="127.0.0.1"))
    client._send_payload = lambda payload: None
    client._read_response = lambda: {"actpos": 425}

    assert client._read_gripper_value() == 0.425


def test_socket_client_initializes_modbus_for_register_feedback():
    client = RealManSocketClient(
        RealManClientConfig(
            host="127.0.0.1",
            gripper_position_register=0x0609,
            initialize_gripper_modbus=True,
            initialize_gripper_route=False,
        )
    )
    sent_payloads = []
    client._send_command = sent_payloads.append

    client.configure()

    assert sent_payloads == [{"command": "set_modbus_mode", "port": 1, "baudrate": 115200, "timeout": 2}]


def test_socket_client_does_not_reconfigure_hardware_initialized_gripper():
    client = RealManSocketClient(
        RealManClientConfig(
            host="127.0.0.1",
            gripper_position_register=0x0609,
            initialize_gripper_modbus=False,
            initialize_gripper_route=False,
        )
    )
    sent_payloads = []
    client._send_command = sent_payloads.append

    client.configure()

    assert sent_payloads == []


def test_socket_client_scales_gripper_command():
    sent_payloads = []

    client = RealManSocketClient(
        RealManClientConfig(host="127.0.0.1", has_gripper=True, gripper_scale=1000.0)
    )
    client._send_command = sent_payloads.append

    client.send_joint_positions({"gripper": 0.425})

    assert sent_payloads == [{"command": "set_gripper_position", "position": 425, "block": False}]


def test_socket_client_reports_required_gripper_read_failures():
    client = RealManSocketClient(
        RealManClientConfig(host="127.0.0.1", has_gripper=True, gripper_scale=1000.0)
    )

    def fail_read(payload, expected_key):
        raise TimeoutError("no gripper response")

    client._send_command_with_expected_key = fail_read

    try:
        client._read_gripper_value()
    except RuntimeError as exc:
        message = str(exc)
        assert "gripper is required" in message
        assert "get_gripper_state" in message
        assert "no gripper response" in message
    else:
        raise AssertionError("Expected required gripper read failure")


def test_socket_client_reports_required_gripper_write_failures():
    client = RealManSocketClient(
        RealManClientConfig(host="127.0.0.1", has_gripper=True, gripper_scale=1000.0)
    )

    def fail_send(payload):
        raise TimeoutError("no gripper command response")

    client._send_command = fail_send

    try:
        client.send_joint_positions({"gripper": 0.425})
    except RuntimeError as exc:
        message = str(exc)
        assert "gripper is required" in message
        assert "set_gripper_position" in message
        assert "no gripper command response" in message
    else:
        raise AssertionError("Expected required gripper write failure")


def test_socket_client_pops_newline_delimited_json_messages():
    client = RealManSocketClient(RealManClientConfig(host="127.0.0.1"))
    client._receive_buffer = '{"state":"moving"}\r\n{"joint":[1,2,3,4,5,6]}\r\n'

    assert client._pop_buffered_line() == '{"state":"moving"}'
    assert client._pop_buffered_line() == '{"joint":[1,2,3,4,5,6]}'
    assert client._pop_buffered_line() is None


def test_sdk_client_excludes_gripper_from_arm_joint_command():
    class FakeRobot:
        def __init__(self):
            self.last_movej = None
            self.last_gripper = None

        def rm_movej_canfd(self, joints, follow):
            self.last_movej = (joints, follow)
            return 0

        def rm_set_gripper_position(self, position, speed, block):
            self.last_gripper = (position, speed, block)
            return 0

    config = RealManClientConfig(host="127.0.0.1", has_gripper=True, gripper_scale=10.0)
    robot = FakeRobot()
    client = object.__new__(RealManSDKClient)
    BaseRealManClient.__init__(client, config)
    client._robot = robot
    client._handle = object()
    client._is_connected = True

    action = {f"joint_{idx}": float(idx) for idx in range(1, 7)}
    action["gripper"] = 42.0

    client.send_joint_positions(action)

    assert robot.last_movej == ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], False)
    assert robot.last_gripper == (420, 500, False)


def test_sdk_client_adds_gripper_placeholder_when_absent():
    class FakeRobot:
        def rm_get_current_arm_joint(self):
            return 0, {"joint": [1, 2, 3, 4, 5, 6]}

    config = RealManClientConfig(host="127.0.0.1", has_gripper=False)
    client = object.__new__(RealManSDKClient)
    BaseRealManClient.__init__(client, config)
    client._robot = FakeRobot()
    client._handle = object()
    client._is_connected = True

    assert client.get_joint_positions()["gripper"] == 0.0


def test_rm_aloha_master_scales_all_joints_when_gripper_is_not_configured():
    config = RMAlohaMasterConfig(
        port="/dev/null",
        joint_names=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"),
        joint_scale=10000.0,
    )
    teleop = object.__new__(RMAlohaMaster)
    teleop.config = config

    frame = bytearray.fromhex("55AA0200000000")
    for raw_value in (10000, 20000, 30000, 40000, 50000, 60000, 70000):
        frame.extend(raw_value.to_bytes(4, byteorder="little", signed=True))
        frame.extend(b"\x00")

    assert teleop._decode_master_state(frame.hex().upper()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_rm_aloha_master_does_not_treat_headerless_tail_as_complete_frame():
    config = RMAlohaMasterConfig(
        port="/dev/null",
        frame_header="AA55",
        state_frame_marker="0200002301",
        state_frame_fallback_marker="",
        min_frame_hex_length=82,
    )
    teleop = object.__new__(RMAlohaMaster)
    teleop.config = config

    assert not teleop._has_complete_frame(
        "52AB09000023017D4AFDFF0152DC0500011A3B0C0001F101FFFF0154C5080001846C0600016C0000006B"
    )


def test_rm_aloha_master_can_fall_back_to_state_frame_marker():
    config = RMAlohaMasterConfig(
        port="/dev/null",
        frame_header="AA55",
        min_frame_hex_length=82,
        gripper_scale=1000.0,
    )
    teleop = object.__new__(RMAlohaMaster)
    teleop.config = config

    frame = bytearray.fromhex("54550200002301")
    for raw_value in (10000, 20000, 30000, 40000, 50000, 60000, 700):
        frame.extend(raw_value.to_bytes(4, byteorder="little", signed=True))
        frame.extend(b"\x01")

    assert teleop._has_complete_frame(frame.hex().upper())
    assert teleop._decode_master_state(frame.hex().upper()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7]


def test_rm_aloha_master_scales_gripper():
    config = RMAlohaMasterConfig(
        port="/dev/null",
        frame_header="AA55",
        min_frame_hex_length=82,
        gripper_scale=1000.0,
    )
    teleop = object.__new__(RMAlohaMaster)
    teleop.config = config

    encoded = "A4AB09000023017D4AFDFF0152DC0500011A3B0C0001F101FFFF0154C5080001846C0600016C0000006B"

    assert teleop._has_complete_frame(encoded)
    assert teleop._decode_master_state(encoded) == [
        -17.7539,
        38.4082,
        80.1562,
        -6.5039,
        57.4804,
        42.0996,
        0.108,
    ]


def test_rm_aloha_sdk_client_uses_factory_sdk_method_names():
    class GripperState:
        actpos = 123

    class FakeArm:
        def __init__(self):
            self.last_canfd = None
            self.last_gripper = None

        def Get_Joint_Degree(self):  # noqa: N802
            return 0, [1, 2, 3, 4, 5, 6, 999]

        def Get_Gripper_State(self):  # noqa: N802
            return 0, GripperState()

        def Movej_CANFD(self, joints, follow):  # noqa: N802
            self.last_canfd = (joints, follow)
            return 0

        def Set_Gripper_Position(self, position, block):  # noqa: N802
            self.last_gripper = (position, block)
            return 0

    config = RealManClientConfig(host="127.0.0.1", has_gripper=True, gripper_scale=1000.0)
    arm = FakeArm()
    client = object.__new__(RealManAlohaSDKClient)
    BaseRealManClient.__init__(client, config)
    client._arm = arm
    client._is_connected = True

    assert client.get_joint_positions() == {
        "joint_1": 1.0,
        "joint_2": 2.0,
        "joint_3": 3.0,
        "joint_4": 4.0,
        "joint_5": 5.0,
        "joint_6": 6.0,
        "gripper": 0.123,
    }

    action = {f"joint_{idx}": float(idx) for idx in range(1, 7)}
    action["gripper"] = 0.042
    client.send_joint_positions(action)

    assert arm.last_canfd == ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], False)
    assert arm.last_gripper == (42, False)


def test_rm_aloha_sdk_client_uses_configured_gripper_state_field():
    class GripperState:
        actpos = 0
        position = 456

    class FakeArm:
        def Get_Joint_Degree(self):  # noqa: N802
            return 0, [1, 2, 3, 4, 5, 6, 999]

        def Get_Gripper_State(self):  # noqa: N802
            return 0, GripperState()

    config = RealManClientConfig(
        host="127.0.0.1",
        has_gripper=True,
        gripper_scale=1000.0,
        gripper_state_position_key="position",
    )
    arm = FakeArm()
    client = object.__new__(RealManAlohaSDKClient)
    BaseRealManClient.__init__(client, config)
    client._arm = arm
    client._is_connected = True

    assert client.get_joint_positions()["gripper"] == 0.456


def test_rm_aloha_sdk_client_passes_configured_port_to_arm():
    class FakeArm:
        calls = []

        def __init__(self, arm_model, host, port):
            self.calls.append((arm_model, host, port))

        def Arm_Socket_State(self):  # noqa: N802
            return 0

    class FakeSDK:
        Arm = FakeArm

    config = RealManClientConfig(host="127.0.0.1", port=9090, arm_model=65)
    client = object.__new__(RealManAlohaSDKClient)
    BaseRealManClient.__init__(client, config)
    client._sdk = FakeSDK()
    client._arm = None

    client.connect()

    assert FakeArm.calls == [(65, "127.0.0.1", 9090)]

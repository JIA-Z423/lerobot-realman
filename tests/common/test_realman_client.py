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

from lerobot.common.realman.client import (
    BaseRealManClient,
    RealManAlohaSDKClient,
    RealManClientConfig,
    RealManSDKClient,
    RealManSocketClient,
)
from lerobot.teleoperators.rm_aloha_master.config_rm_aloha_master import RMAlohaMasterConfig
from lerobot.teleoperators.rm_aloha_master.rm_aloha_master import RMAlohaMaster


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

    assert not teleop._has_complete_frame("52AB09000023017D4AFDFF0152DC0500011A3B0C0001F101FFFF0154C5080001846C0600016C0000006B")


def test_rm_aloha_master_can_fall_back_to_state_frame_marker():
    config = RMAlohaMasterConfig(port="/dev/null", frame_header="AA55", min_frame_hex_length=82)
    teleop = object.__new__(RMAlohaMaster)
    teleop.config = config

    frame = bytearray.fromhex("54550200002301")
    for raw_value in (10000, 20000, 30000, 40000, 50000, 60000, 70000):
        frame.extend(raw_value.to_bytes(4, byteorder="little", signed=True))
        frame.extend(b"\x01")

    assert teleop._has_complete_frame(frame.hex().upper())
    assert teleop._decode_master_state(frame.hex().upper()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 70000.0]


def test_rm_aloha_master_accepts_ab_variant_state_frame():
    config = RMAlohaMasterConfig(port="/dev/null", frame_header="AA55", min_frame_hex_length=82)
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
        108.0,
    ]


def test_rm_aloha_sdk_client_uses_factory_sdk_method_names():
    class GripperState:
        actpos = 123

    class FakeArm:
        def __init__(self):
            self.last_canfd = None
            self.last_gripper = None

        def Get_Joint_Degree(self):
            return 0, [1, 2, 3, 4, 5, 6, 999]

        def Get_Gripper_State(self):
            return 0, GripperState()

        def Movej_CANFD(self, joints, follow):
            self.last_canfd = (joints, follow)
            return 0

        def Set_Gripper_Position(self, position, block):
            self.last_gripper = (position, block)
            return 0

    config = RealManClientConfig(host="127.0.0.1", has_gripper=True, gripper_scale=1.0)
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
        "gripper": 123.0,
    }

    action = {f"joint_{idx}": float(idx) for idx in range(1, 7)}
    action["gripper"] = 42.0
    client.send_joint_positions(action)

    assert arm.last_canfd == ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], False)
    assert arm.last_gripper == (42, False)

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

"""Configuration for RealMan follower connections, feedback, and cameras."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.common.realman import RealManControlMode

from ..config import RobotConfig


@dataclass
class RealManFollowerBaseConfig:
    """Shared hardware settings for single-arm and double-arm RealMan followers."""

    host: str
    port: int | None = None
    arm_id: str | None = None
    joint_names: tuple[str, ...] = (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
    use_degrees: bool = False
    control_hz: int = 30
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 1.0
    control_mode: RealManControlMode = RealManControlMode.JOINT_POSITION
    joint_scale: float = 1000.0
    has_gripper: bool = True
    gripper_scale: float = 1000.0
    gripper_feedback_source: str = "native"
    gripper_state_position_key: str = "actpos"
    gripper_position_register: int | None = None
    gripper_modbus_port: int = 1
    gripper_modbus_device: int = 1
    gripper_modbus_baudrate: int = 115200
    gripper_modbus_timeout: int = 2
    initialize_gripper_modbus: bool = False
    gripper_open_value: int = 1000
    gripper_speed: int = 500
    initialize_gripper_route: bool = True
    move_command: str = "movej_canfd"
    max_relative_target: float | dict[str, float] | None = None
    enable_calibration: bool = False
    sdk_backend: str | None = "socket_json"
    arm_model: int = 65
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("realman_follower")
@dataclass
class RealManFollowerConfig(RobotConfig, RealManFollowerBaseConfig):
    """Configure a RealMan follower arm and its attached cameras.

    Passive recording requires `initialize_gripper_route`, `initialize_gripper_modbus`, and
    `enable_calibration` to be `False`. RM+ feedback is read-only and requires the `socket_json` backend.

    Args:
        host (`str`):
            Hostname or IP address of the arm controller.
        port (`int | None`, *optional*):
            Controller TCP port. The transport uses 8080 when omitted; this is not a serial-device path.
        arm_id (`str | None`, *optional*):
            Arm identifier retained for transport compatibility; the current backends do not use it.
        joint_names (`tuple[str, ...]`, *optional*, defaults to `('joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper')`):
            Ordered joint names used to construct action and observation features.
        use_degrees (`bool`, *optional*, defaults to `False`):
            Compatibility option; current backends do not use it to convert joint units.
        control_hz (`int`, *optional*, defaults to 30):
            Compatibility frequency. Passive recording samples at `dataset.fps` and does not run a
            master-follower control loop.
        connect_timeout_s (`float`, *optional*, defaults to 5.0):
            TCP connection timeout in seconds for the socket backend.
        read_timeout_s (`float`, *optional*, defaults to 1.0):
            Socket response timeout in seconds.
        control_mode (`RealManControlMode`, *optional*, defaults to `RealManControlMode.JOINT_POSITION`):
            Control-mode setting retained for transport compatibility.
        joint_scale (`float`, *optional*, defaults to 1000.0):
            Divisor for socket joint feedback and multiplier for socket joint targets.
        has_gripper (`bool`, *optional*, defaults to `True`):
            Whether to read physical gripper feedback when the gripper is included in `joint_names`.
        gripper_scale (`float`, *optional*, defaults to 1000.0):
            Divisor for raw gripper positions. RM+ values divided by 1000 are normalized, not millimeters.
        gripper_feedback_source (`str`, *optional*, defaults to `"native"`):
            Feedback source, either `"native"` or `"rm_plus"`. RM+ rejects action sending and cannot be
            combined with `gripper_position_register`.
        gripper_state_position_key (`str`, *optional*, defaults to `"actpos"`):
            Position field in native gripper-state responses.
        gripper_position_register (`int | None`, *optional*):
            First Modbus register for optional native gripper position feedback on the socket backend.
        gripper_modbus_port (`int`, *optional*, defaults to 1):
            Controller Modbus port selector for register feedback.
        gripper_modbus_device (`int`, *optional*, defaults to 1):
            Modbus device address for register feedback.
        gripper_modbus_baudrate (`int`, *optional*, defaults to 115200):
            Baud rate used when explicitly initializing the controller's Modbus link.
        gripper_modbus_timeout (`int`, *optional*, defaults to 2):
            Controller timeout parameter passed to `set_modbus_mode` during initialization.
        initialize_gripper_modbus (`bool`, *optional*, defaults to `False`):
            Initialize the Modbus link during configuration. Keep disabled for passive recording and RM+.
        gripper_open_value (`int`, *optional*, defaults to 1000):
            Raw upper position limit used by gripper-route initialization.
        gripper_speed (`int`, *optional*, defaults to 500):
            Speed parameter for supported SDK gripper target commands.
        initialize_gripper_route (`bool`, *optional*, defaults to `True`):
            Initialize the native gripper route on connection. Set to `False` for passive recording or RM+.
        move_command (`str`, *optional*, defaults to `"movej_canfd"`):
            Socket JSON command used for joint targets outside passive recording.
        max_relative_target (`float | dict[str, float] | None`, *optional*):
            Maximum joint-target change relative to current feedback, shared or keyed by joint name.
        enable_calibration (`bool`, *optional*, defaults to `False`):
            Enable the LeRobot calibration workflow. Keep disabled for passive recording; calibration is
            skipped by default.
        sdk_backend (`str | None`, *optional*, defaults to `"socket_json"`):
            Transport backend: `"socket_json"`, `"rm_aloha_sdk"`, or `"vendor_sdk"`. `None` selects sockets.
        arm_model (`int`, *optional*, defaults to 65):
            Arm model identifier used by the bundled RM ALOHA SDK backend.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Attached camera configurations, keyed by observation name. Defaults to an empty dictionary.
        id (`str | None`, *optional*):
            Identifier used to distinguish this arm and name its calibration file.
        calibration_dir (`Path | None`, *optional*):
            Directory for LeRobot calibration files. When omitted, the robot uses its default cache path.
    """

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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.common.realman import RealManControlMode

from ..config import RobotConfig


@dataclass
class RealManFollowerBaseConfig:
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
    pass

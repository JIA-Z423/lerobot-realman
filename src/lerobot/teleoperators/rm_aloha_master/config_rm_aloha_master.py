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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class RMAlohaMasterBaseConfig:
    port: str
    baud_rate: int = 460800
    timeout_s: float = 0.0
    frame_timeout_s: float = 0.1
    read_size: int = 128
    frame_header: str = "55AA"
    state_frame_marker: str = "0200002301"
    state_frame_fallback_marker: str = "00002301"
    min_frame_hex_length: int = 82
    init_hex_command: str = "55 AA 02 00 00 67"
    joint_names: tuple[str, ...] = (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
    joint_scale: float = 10000.0


@TeleoperatorConfig.register_subclass("rm_aloha_master")
@dataclass
class RMAlohaMasterConfig(TeleoperatorConfig, RMAlohaMasterBaseConfig):
    pass

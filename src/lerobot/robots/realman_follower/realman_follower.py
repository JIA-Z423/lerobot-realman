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

import logging
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.common.realman import RealManClientConfig, RealManDeviceRole, make_realman_client
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_realman_follower import RealManFollowerConfig

logger = logging.getLogger(__name__)


class RealManFollower(Robot):
    config_class = RealManFollowerConfig
    name = "realman_follower"

    def __init__(self, config: RealManFollowerConfig):
        super().__init__(config)
        self.config = config
        self.client = make_realman_client(
            RealManClientConfig(
                host=config.host,
                port=config.port,
                arm_id=config.arm_id,
                use_degrees=config.use_degrees,
                joint_names=config.joint_names,
                connect_timeout_s=config.connect_timeout_s,
                read_timeout_s=config.read_timeout_s,
                control_hz=config.control_hz,
                control_mode=config.control_mode,
                role=RealManDeviceRole.FOLLOWER,
                sdk_backend=config.sdk_backend,
                arm_model=config.arm_model,
                joint_scale=config.joint_scale,
                has_gripper=config.has_gripper,
                gripper_scale=config.gripper_scale,
                gripper_open_value=config.gripper_open_value,
                gripper_speed=config.gripper_speed,
                initialize_gripper_route=config.initialize_gripper_route,
                move_command=config.move_command,
            )
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.client.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self.client.connect()
            for cam in self.cameras.values():
                cam.connect()
            self.configure()
            if calibrate and not self.is_calibrated and self.config.enable_calibration:
                self.calibrate()
        except Exception:
            self.client.disconnect()
            for cam in self.cameras.values():
                if cam.is_connected:
                    cam.disconnect()
            raise
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return not self.config.enable_calibration or bool(self.calibration)

    def calibrate(self) -> None:
        if not self.config.enable_calibration:
            logger.info("%s does not require LeRobot-side calibration.", self)
            return
        self.client.calibrate()
        self._save_calibration()

    def configure(self) -> None:
        self.client.configure()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = {f"{joint}.pos": value for joint, value in self.client.get_joint_positions().items()}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {
            key.removesuffix(".pos"): float(val) for key, val in action.items() if key.endswith(".pos")
        }

        if self.config.max_relative_target is not None:
            present_pos = self.client.get_joint_positions()
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        sent = self.client.send_joint_positions(goal_pos)
        return {f"{joint}.pos": value for joint, value in sent.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self.client.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)

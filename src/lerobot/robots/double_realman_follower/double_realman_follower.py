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

from concurrent.futures import ThreadPoolExecutor, wait
from functools import cached_property

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..realman_follower import RealManFollower, RealManFollowerConfig
from ..robot import Robot
from .config_double_realman_follower import DoubleRealManFollowerConfig


class DoubleRealManFollower(Robot):
    config_class = DoubleRealManFollowerConfig
    name = "double_realman_follower"

    def __init__(self, config: DoubleRealManFollowerConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = RealManFollower(
            RealManFollowerConfig(
                id=f"{config.id}_left" if config.id else None,
                calibration_dir=config.calibration_dir,
                **config.left_arm_config.__dict__,
            )
        )
        self.right_arm = RealManFollower(
            RealManFollowerConfig(
                id=f"{config.id}_right" if config.id else None,
                calibration_dir=config.calibration_dir,
                **config.right_arm_config.__dict__,
            )
        )
        self.cameras = {
            **{f"left_{key}": camera for key, camera in self.left_arm.cameras.items()},
            **{f"right_{key}": camera for key, camera in self.right_arm.cameras.items()},
        }
        self._executor: ThreadPoolExecutor | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.observation_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.observation_features.items()},
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.action_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.action_features.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self.left_arm.connect(calibrate)
            self.right_arm.connect(calibrate)
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="realman_double_arm")
        except Exception as exc:
            cleanup_errors = self._disconnect_arms()
            if cleanup_errors:
                raise ExceptionGroup(
                    "RealMan connection and cleanup failed", [exc, *cleanup_errors]
                ) from None
            raise

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        assert self._executor is not None
        left_future = self._executor.submit(self.left_arm.get_observation)
        right_future = self._executor.submit(self.right_arm.get_observation)
        # Finish both reads before propagating either error, so cleanup cannot close a busy connection.
        wait((left_future, right_future))
        left_obs = left_future.result()
        right_obs = right_future.result()
        return {
            **{f"left_{key}": value for key, value in left_obs.items()},
            **{f"right_{key}": value for key, value in right_obs.items()},
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self._executor is not None
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        left_future = self._executor.submit(self.left_arm.send_action, left_action)
        right_future = self._executor.submit(self.right_arm.send_action, right_action)
        wait((left_future, right_future))
        sent_left = left_future.result()
        sent_right = right_future.result()
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    def disconnect(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        errors = self._disconnect_arms()
        if errors:
            raise ExceptionGroup("Failed to disconnect RealMan arms", errors)

    def _disconnect_arms(self) -> list[Exception]:
        errors = []
        for arm in (self.left_arm, self.right_arm):
            if arm.is_connected:
                try:
                    arm.disconnect()
                except Exception as exc:
                    errors.append(exc)
        return errors

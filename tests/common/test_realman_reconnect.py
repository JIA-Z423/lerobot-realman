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

from threading import Event, Thread

import pytest

from lerobot.robots.double_realman_follower import (
    DoubleRealManFollowerConfig,
    double_realman_follower as follower_module,
)
from lerobot.robots.realman_follower import RealManFollowerBaseConfig
from lerobot.teleoperators.double_rm_aloha_master import (
    DoubleRMAlohaMasterConfig,
    double_rm_aloha_master as master_module,
)
from lerobot.teleoperators.rm_aloha_master import RMAlohaMasterBaseConfig


class FakeArm:
    """In-memory arm used in place of both hardware adapter constructors."""

    def __init__(self, config):
        self.id = config.id
        self.is_connected = False
        self.cameras = {}
        self.connect_count = 0
        self.disconnect_count = 0
        self.fail_connect = False
        self.fail_disconnect = False
        self.fail_read = False
        self.read_started = Event()
        self.read_finished = Event()
        self.release_read = None

    def connect(self, calibrate=True):
        self.connect_count += 1
        self.is_connected = True
        if self.fail_connect:
            raise RuntimeError(f"connect failed: {self.id}")

    def disconnect(self):
        self.disconnect_count += 1
        assert not self.read_started.is_set() or self.read_finished.is_set()
        self.is_connected = False
        if self.fail_disconnect:
            raise RuntimeError(f"disconnect failed: {self.id}")

    def _read(self):
        self.read_started.set()
        try:
            if self.release_read is not None:
                assert self.release_read.wait(timeout=5), "Test failed to release the simulated read"
            if self.fail_read:
                raise RuntimeError(f"read failed: {self.id}")
            return {"gripper.pos": 0.25 if self.id.endswith("_left") else 0.75}
        finally:
            self.read_finished.set()

    def get_observation(self):
        return self._read()

    def get_action(self):
        return self._read()

    def send_action(self, action):
        self._read()
        return action


@pytest.fixture(params=["follower", "master"])
def double_device(request, monkeypatch, tmp_path):
    if request.param == "follower":
        monkeypatch.setattr(follower_module, "RealManFollower", FakeArm)
        config = DoubleRealManFollowerConfig(
            id="offline",
            calibration_dir=tmp_path,
            left_arm_config=RealManFollowerBaseConfig(host="unused-left"),
            right_arm_config=RealManFollowerBaseConfig(host="unused-right"),
        )
        device = follower_module.DoubleRealManFollower(config)
        read_method = "get_observation"
    else:
        monkeypatch.setattr(master_module, "RMAlohaMaster", FakeArm)
        config = DoubleRMAlohaMasterConfig(
            id="offline",
            calibration_dir=tmp_path,
            left_arm_config=RMAlohaMasterBaseConfig(port="unused-left"),
            right_arm_config=RMAlohaMasterBaseConfig(port="unused-right"),
        )
        device = master_module.DoubleRMAlohaMaster(config)
        read_method = "get_action"
    try:
        yield device, read_method
    finally:
        device.left_arm.fail_disconnect = False
        device.right_arm.fail_disconnect = False
        device.disconnect()


def test_double_realman_can_reconnect_and_read_same_object(double_device):
    device, read_method = double_device
    previous_executor = None
    for _ in range(2):
        device.connect(calibrate=False)
        assert device._executor is not previous_executor
        previous_executor = device._executor
        assert getattr(device, read_method)() == {"left_gripper.pos": 0.25, "right_gripper.pos": 0.75}
        device.disconnect()
        assert not device.left_arm.is_connected
        assert not device.right_arm.is_connected
        assert device._executor is None

    assert device.left_arm.connect_count == device.right_arm.connect_count == 2
    assert device.left_arm.disconnect_count == device.right_arm.disconnect_count == 2


def test_double_realman_waits_for_other_read_before_raising(double_device):
    device, read_method = double_device
    _assert_parallel_error_waits_for_other_arm(device, lambda: getattr(device, read_method)())


def test_double_realman_waits_for_other_send_before_raising(double_device):
    device, _ = double_device
    if not isinstance(device, follower_module.DoubleRealManFollower):
        pytest.skip("Only follower arms send actions")
    _assert_parallel_error_waits_for_other_arm(device, lambda: device.send_action({}))


def _assert_parallel_error_waits_for_other_arm(device, operation):
    device.connect(calibrate=False)
    device.left_arm.fail_read = True
    device.right_arm.release_read = Event()
    operation_finished = Event()
    errors = []

    def run_operation():
        try:
            operation()
        except Exception as exc:
            errors.append(exc)
        finally:
            operation_finished.set()

    thread = Thread(target=run_operation)
    thread.start()
    try:
        assert device.right_arm.read_started.wait(timeout=2)
        assert not operation_finished.wait(timeout=0.05)
    finally:
        device.right_arm.release_read.set()
        thread.join(timeout=2)

    assert operation_finished.is_set()
    assert len(errors) == 1
    assert str(errors[0]) == "read failed: offline_left"
    assert device.right_arm.read_finished.is_set()
    device.disconnect()


def test_double_realman_disconnect_attempts_both_arms_after_error(double_device):
    device, _ = double_device
    device.connect(calibrate=False)
    device.left_arm.fail_disconnect = device.right_arm.fail_disconnect = True

    with pytest.raises(ExceptionGroup) as exc_info:
        device.disconnect()

    assert len(exc_info.value.exceptions) == 2
    assert device.left_arm.disconnect_count == device.right_arm.disconnect_count == 1
    assert device._executor is None


def test_double_realman_cleans_up_connection_failure_then_reconnects(double_device):
    device, read_method = double_device
    device.right_arm.fail_connect = True

    with pytest.raises(RuntimeError, match="connect failed: offline_right"):
        device.connect(calibrate=False)

    assert device.left_arm.disconnect_count == device.right_arm.disconnect_count == 1
    assert not device.left_arm.is_connected
    assert not device.right_arm.is_connected
    device.right_arm.fail_connect = False
    device.connect(calibrate=False)
    assert len(getattr(device, read_method)()) == 2
    device.disconnect()


def test_double_realman_connection_failure_attempts_both_cleanups(double_device):
    device, _ = double_device
    device.right_arm.fail_connect = True
    device.left_arm.fail_disconnect = device.right_arm.fail_disconnect = True

    with pytest.raises(ExceptionGroup) as exc_info:
        device.connect(calibrate=False)

    assert len(exc_info.value.exceptions) == 3
    assert "connect failed: offline_right" in str(exc_info.value.exceptions[0])
    assert device.left_arm.disconnect_count == device.right_arm.disconnect_count == 1


def test_double_realman_disconnect_cleans_up_partially_connected_device(double_device):
    device, _ = double_device
    device.connect(calibrate=False)
    device.left_arm.is_connected = False

    device.disconnect()

    assert device.left_arm.disconnect_count == 0
    assert device.right_arm.disconnect_count == 1
    assert device._executor is None

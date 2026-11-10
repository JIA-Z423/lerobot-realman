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

from unittest.mock import Mock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.robots.double_realman_follower import DoubleRealManFollowerConfig
from lerobot.robots.realman_follower import RealManFollowerBaseConfig, RealManFollowerConfig
from lerobot.scripts import lerobot_record
from lerobot.teleoperators.rm_aloha_master import RMAlohaMasterConfig
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


@pytest.mark.parametrize("arm_path", ["robot", "robot.left_arm_config", "robot.right_arm_config"])
@pytest.mark.parametrize(
    "option", ["initialize_gripper_route", "initialize_gripper_modbus", "enable_calibration"]
)
def test_passive_recording_rejects_initialization_before_creating_devices(monkeypatch, arm_path, option):
    if arm_path == "robot":
        robot_config = RealManFollowerConfig(host="unused", initialize_gripper_route=False)
        arm_config = robot_config
    else:
        robot_config = DoubleRealManFollowerConfig(
            left_arm_config=RealManFollowerBaseConfig(host="unused-left", initialize_gripper_route=False),
            right_arm_config=RealManFollowerBaseConfig(host="unused-right", initialize_gripper_route=False),
        )
        arm_config = getattr(robot_config, arm_path.removeprefix("robot."))
    setattr(arm_config, option, True)
    cfg = lerobot_record.RecordConfig(
        robot=robot_config,
        dataset=lerobot_record.DatasetRecordConfig(repo_id="local/unused", single_task="offline test"),
        teleop=RMAlohaMasterConfig(port="unused"),
        passive_recording=True,
    )
    make_robot = Mock(side_effect=AssertionError("Must not create or connect devices"))
    make_teleop = Mock(side_effect=AssertionError("Must not create or connect devices"))
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", make_robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", make_teleop)

    with pytest.raises(ValueError, match=rf"{arm_path}\.{option}=true"):
        lerobot_record.record(cfg)

    make_robot.assert_not_called()
    make_teleop.assert_not_called()
    assert getattr(arm_config, option) is True


def test_passive_recording_accepts_read_only_realman_configs():
    robot_config = DoubleRealManFollowerConfig(
        left_arm_config=RealManFollowerBaseConfig(host="unused-left", initialize_gripper_route=False),
        right_arm_config=RealManFollowerBaseConfig(host="unused-right", initialize_gripper_route=False),
    )

    lerobot_record._validate_realman_passive_recording_config(robot_config)
    lerobot_record._validate_realman_passive_recording_config(
        RealManFollowerConfig(host="unused", initialize_gripper_route=False)
    )


def test_active_recording_preserves_explicit_initialization_options(monkeypatch):
    cfg = lerobot_record.RecordConfig(
        robot=RealManFollowerConfig(host="unused", initialize_gripper_route=True),
        dataset=lerobot_record.DatasetRecordConfig(repo_id="local/unused", single_task="offline test"),
        teleop=RMAlohaMasterConfig(port="unused"),
        passive_recording=False,
    )
    monkeypatch.setattr(lerobot_record, "init_logging", lambda: None)
    make_robot = Mock(side_effect=RuntimeError("offline factory sentinel"))
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", make_robot)

    with pytest.raises(RuntimeError, match="offline factory sentinel"):
        lerobot_record.record(cfg)

    make_robot.assert_called_once_with(cfg.robot)
    assert cfg.robot.initialize_gripper_route is True


def test_passive_recording_records_and_resets_without_sending_actions(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    cfg = lerobot_record.RecordConfig(
        robot=MockRobotConfig(
            calibration_dir=tmp_path / "robot_calibration",
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        ),
        teleop=MockTeleopConfig(
            calibration_dir=tmp_path / "teleop_calibration",
            random_values=False,
            static_values=[4.0, 5.0, 6.0],
        ),
        dataset=lerobot_record.DatasetRecordConfig(
            repo_id="local/passive_recording",
            single_task="offline passive recording",
            root=tmp_path / "record",
            fps=30,
            num_episodes=2,
            episode_time_s=0.1,
            reset_time_s=0.1,
            video=False,
            push_to_hub=False,
            no_stamp=True,
        ),
        passive_recording=True,
        play_sounds=False,
    )
    send_action = Mock(side_effect=AssertionError("Passive recording must not send actions"))
    monkeypatch.setattr(MockRobot, "send_action", send_action)
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda: (None, events))
    original_get_action = MockTeleop.get_action
    action_reads = []

    def get_action(teleop):
        action = original_get_action(teleop)
        action_reads.append(action)
        return action

    monkeypatch.setattr(MockTeleop, "get_action", get_action)

    dataset = lerobot_record.record(cfg)

    send_action.assert_not_called()
    assert dataset.repo_id == "local/passive_recording"
    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    rows = [
        row
        for path in sorted((cfg.dataset.root / "data").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]
    assert len(rows) == dataset.meta.total_frames == dataset.num_frames
    assert {row["episode_index"] for row in rows} == {0, 1}
    for episode_index in (0, 1):
        episode = [row for row in rows if row["episode_index"] == episode_index]
        assert [row["frame_index"] for row in episode] == list(range(len(episode)))
    for row in rows:
        assert row["observation.state"] == [1.0, 2.0, 3.0]
        assert row["action"] == [4.0, 5.0, 6.0]
    # The reset phase reads the teleoperator too, but does not write dataset frames.
    assert len(action_reads) > len(rows)

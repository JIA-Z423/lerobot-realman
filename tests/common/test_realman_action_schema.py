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

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.scripts.lerobot_record import _validate_passive_recording_action_features
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def test_passive_recording_requires_matching_robot_and_teleop_action_features():
    robot = MockRobotConfig(n_motors=3)
    teleop = MockTeleopConfig(n_motors=2)

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    with pytest.raises(ValueError) as exc_info:
        _validate_passive_recording_action_features(
            make_robot_from_config(robot), make_teleoperator_from_config(teleop)
        )

    message = str(exc_info.value)
    assert "passive_recording=true" in message
    assert "Missing from teleoperator: ['motor_3.pos']" in message
    assert "Extra from teleoperator: []" in message

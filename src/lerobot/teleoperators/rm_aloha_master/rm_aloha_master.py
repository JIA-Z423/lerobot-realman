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

import binascii
import logging
import time
from typing import TYPE_CHECKING, Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]

from ..teleoperator import Teleoperator
from .config_rm_aloha_master import RMAlohaMasterConfig

logger = logging.getLogger(__name__)


class RMAlohaMaster(Teleoperator):
    config_class = RMAlohaMasterConfig
    name = "rm_aloha_master"

    _field_slices = (
        (14, 22),
        (24, 32),
        (34, 42),
        (44, 52),
        (54, 62),
        (64, 72),
        (74, 82),
    )

    def __init__(self, config: RMAlohaMasterConfig):
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        super().__init__(config)
        self.config = config
        if len(config.joint_names) > len(self._field_slices):
            raise ValueError(
                f"RM Aloha master supports at most {len(self._field_slices)} joints, "
                f"got {len(config.joint_names)}."
            )
        self.serial = serial.Serial(port=None, baudrate=config.baud_rate, timeout=config.timeout_s)
        self.serial.port = config.port
        self._init_bytes = binascii.unhexlify(config.init_hex_command.replace(" ", ""))

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joint_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.serial.is_open

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not self.serial.is_open:
            self.serial.open()
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        self.serial.write(self._init_bytes)
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s does not require calibration.", self)

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self.serial.write(self._init_bytes)
        encoded = self._read_encoded_frame()
        values = self._decode_master_state(encoded)
        return {f"{joint}.pos": value for joint, value in zip(self.config.joint_names, values, strict=False)}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise NotImplementedError("RM Aloha master does not expose force feedback.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.serial.close()
        logger.info("%s disconnected.", self)

    def _decode_master_state(self, encoded: str) -> list[float]:
        encoded = self._extract_state_frame(encoded)
        if len(encoded) < self._field_slices[-1][1]:
            raise RuntimeError(f"Incomplete RM Aloha master frame on {self.config.port}: {encoded!r}")

        values: list[float] = []
        for index, (start, end) in enumerate(self._field_slices[: len(self.config.joint_names)]):
            joint_name = self.config.joint_names[index]
            field_hex = encoded[start:end]
            raw_value = int.from_bytes(bytearray.fromhex(field_hex), byteorder="little", signed=True)
            if joint_name == "gripper":
                values.append(raw_value / self.config.gripper_scale)
            else:
                values.append(raw_value / self.config.joint_scale)
        return values

    def _read_encoded_frame(self) -> str:
        deadline = time.perf_counter() + self.config.frame_timeout_s
        chunks = bytearray()

        while time.perf_counter() < deadline:
            waiting = self.serial.in_waiting
            read_size = waiting if waiting > 0 else self.config.read_size
            chunk = self.serial.read(read_size)
            if chunk:
                chunks.extend(chunk)
                encoded = binascii.hexlify(chunks).decode("utf-8").upper()
                if self._has_complete_frame(encoded):
                    return encoded
            else:
                time.sleep(0.001)

        encoded = binascii.hexlify(chunks).decode("utf-8").upper()
        raise RuntimeError(f"Timed out reading RM Aloha master frame on {self.config.port}: {encoded!r}")

    def _has_complete_frame(self, encoded: str) -> bool:
        encoded = encoded.replace(" ", "").upper()
        frame = self._extract_state_frame(encoded, allow_incomplete=True)
        return len(frame) >= self.config.min_frame_hex_length

    def _extract_state_frame(self, encoded: str, allow_incomplete: bool = False) -> str:
        encoded = encoded.replace(" ", "").upper()
        start = self._find_state_frame_start(encoded)
        if start < 0:
            if allow_incomplete:
                return ""
            raise RuntimeError(
                f"RM Aloha master state frame not found on {self.config.port}: {encoded!r}"
            )
        return encoded[start:]

    def _find_state_frame_start(self, encoded: str) -> int:
        header = self.config.frame_header.replace(" ", "").upper()
        if header:
            start = encoded.find(header)
            if start >= 0:
                return start

        marker = self.config.state_frame_marker.replace(" ", "").upper()
        marker_start = encoded.find(marker) if marker else -1
        if marker_start >= 4:
            return marker_start - 4

        fallback_marker = self.config.state_frame_fallback_marker.replace(" ", "").upper()
        fallback_start = encoded.find(fallback_marker) if fallback_marker else -1
        if fallback_start >= 6:
            return fallback_start - 6
        return -1

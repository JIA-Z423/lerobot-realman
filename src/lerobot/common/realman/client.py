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

from __future__ import annotations

import importlib
import json
import math
import socket
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, NoReturn


class RealManDeviceRole(str, Enum):  # noqa: UP042 - Preserve the public Enum string representation.
    FOLLOWER = "follower"
    LEADER = "leader"


class RealManControlMode(str, Enum):  # noqa: UP042 - Preserve the public Enum string representation.
    JOINT_POSITION = "joint_position"
    CARTESIAN = "cartesian"


@dataclass(kw_only=True)
class RealManClientConfig:
    host: str
    port: int | None = None
    arm_id: str | None = None
    use_degrees: bool = False
    joint_names: tuple[str, ...] = (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 1.0
    control_hz: int = 30
    control_mode: RealManControlMode = RealManControlMode.JOINT_POSITION
    role: RealManDeviceRole = RealManDeviceRole.FOLLOWER
    sdk_backend: str | None = None
    arm_model: int = 65
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

    def __post_init__(self) -> None:
        if self.gripper_feedback_source not in ("native", "rm_plus"):
            raise ValueError("gripper_feedback_source must be 'native' or 'rm_plus'.")
        if self.gripper_feedback_source == "rm_plus":
            if self.sdk_backend not in (None, "socket_json"):
                raise ValueError("rm_plus gripper feedback is only supported by the socket_json backend.")
            if self.gripper_position_register is not None:
                raise ValueError(
                    "rm_plus gripper feedback cannot be combined with gripper_position_register."
                )
        for name in ("joint_scale", "gripper_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
        address = self.gripper_position_register
        if address is not None:
            if isinstance(address, bool) or not isinstance(address, int) or not 0 <= address <= 0xFFFD:
                raise ValueError("gripper_position_register must be an integer between 0 and 65533.")
            if self.sdk_backend not in (None, "socket_json"):
                raise ValueError("gripper_position_register is only supported by the socket_json backend.")


class BaseRealManClient:
    def __init__(self, config: RealManClientConfig):
        self.config = config
        self._is_connected = False
        self.software_info: dict[str, Any] | None = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def configure(self) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def get_software_info(self) -> dict[str, Any] | None:
        return self.software_info

    def get_joint_positions(self) -> dict[str, float]:
        raise NotImplementedError

    def send_joint_positions(self, positions: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def send_feedback(self, feedback: dict[str, float]) -> None:
        if feedback:
            raise NotImplementedError("Force feedback is not wired for the RealMan integration.")

    def _raise_required_gripper_error(self, operation: str, exc: BaseException | None = None) -> NoReturn:
        message = (
            "RealMan gripper is required by configuration (`has_gripper=True`) but "
            f"`{operation}` failed. Check that the gripper hardware is connected and "
            "the controller exposes the gripper interface."
        )
        if exc is None:
            raise RuntimeError(message)
        raise RuntimeError(f"{message} Original error: {exc}") from exc

    def _get_gripper_state_position(self, payload: Any, operation: str) -> float:
        for name, required_value in (("status", 1), ("error", 0)):
            missing = object()
            value = (
                payload.get(name, missing) if isinstance(payload, dict) else getattr(payload, name, missing)
            )
            if value is not missing and value != required_value:
                self._raise_required_gripper_error(
                    operation,
                    RuntimeError(
                        f"Invalid gripper feedback: {name}={value!r} (expected {required_value}). "
                        f"Position feedback is unavailable; response: {payload!r}"
                    ),
                )
        key = self.config.gripper_state_position_key
        if isinstance(payload, dict):
            raw_value = payload.get(key)
            available = ", ".join(sorted(payload))
        elif isinstance(payload, int | float):
            raw_value = payload
            available = "scalar"
        else:
            raw_value = getattr(payload, key, None)
            available = ", ".join(name for name in dir(payload) if not name.startswith("_"))

        if raw_value is None:
            self._raise_required_gripper_error(
                operation,
                RuntimeError(
                    f"Gripper state field {key!r} was not found in response {payload!r}. "
                    f"Available fields: {available or 'none'}"
                ),
            )
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, int | float)
            or not math.isfinite(raw_value)
        ):
            self._raise_required_gripper_error(
                operation, RuntimeError(f"Gripper position must be a finite number, got {raw_value!r}.")
            )
        position = float(raw_value) / self.config.gripper_scale
        if not math.isfinite(position):
            self._raise_required_gripper_error(
                operation, RuntimeError("Scaled gripper position is not finite.")
            )
        return position


class PlaceholderRealManClient(BaseRealManClient):
    def _unavailable(self) -> RuntimeError:
        return RuntimeError(
            "RealMan transport is not configured. "
            "Use the default socket/json backend or install the vendor SDK and set "
            "`sdk_backend='vendor_sdk'`."
        )

    def connect(self) -> None:
        raise self._unavailable()

    def get_joint_positions(self) -> dict[str, float]:
        raise self._unavailable()

    def send_joint_positions(self, positions: dict[str, float]) -> dict[str, float]:
        raise self._unavailable()


class RealManSocketClient(BaseRealManClient):
    def __init__(self, config: RealManClientConfig):
        super().__init__(config)
        self._socket: socket.socket | None = None
        self._receive_buffer = ""
        self._last_gripper_value = 0.0

    def connect(self) -> None:
        self._socket = socket.create_connection(
            (self.config.host, self.config.port or 8080),
            timeout=self.config.connect_timeout_s,
        )
        self._socket.settimeout(self.config.read_timeout_s)
        self._is_connected = True

    def disconnect(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._receive_buffer = ""
        self._is_connected = False

    def configure(self) -> None:
        if self.config.gripper_feedback_source == "rm_plus":
            if self.config.initialize_gripper_route or self.config.initialize_gripper_modbus:
                raise RuntimeError(
                    "RM+ feedback support is read-only. Set initialize_gripper_route and "
                    "initialize_gripper_modbus to false; no initialization commands were sent."
                )
            return
        if (
            self.config.has_gripper
            and self.config.gripper_position_register is not None
            and self.config.initialize_gripper_modbus
        ):
            try:
                self._send_command(
                    {
                        "command": "set_modbus_mode",
                        "port": self.config.gripper_modbus_port,
                        "baudrate": self.config.gripper_modbus_baudrate,
                        "timeout": self.config.gripper_modbus_timeout,
                    }
                )
            except Exception as exc:
                self._raise_required_gripper_error("set_modbus_mode", exc)
        if self.config.has_gripper and self.config.initialize_gripper_route:
            try:
                self._send_command(
                    {
                        "command": "set_gripper_route",
                        "min": 0,
                        "max": self.config.gripper_open_value,
                    }
                )
            except Exception as exc:
                self._raise_required_gripper_error("set_gripper_route", exc)

    def get_joint_positions(self) -> dict[str, float]:
        joint_values = self._read_joint_values()
        if "gripper" in self.config.joint_names:
            gripper_value = 0.0 if not self.config.has_gripper else self._read_gripper_value()
            joint_values["gripper"] = gripper_value
        return joint_values

    def send_joint_positions(self, positions: dict[str, float]) -> dict[str, float]:
        if self.config.gripper_feedback_source == "rm_plus":
            raise RuntimeError(
                "RM+ feedback support is read-only; sending joint or gripper targets is not supported. "
                "No commands were sent."
            )
        ordered_joint_names = [name for name in self.config.joint_names if name != "gripper"]
        ordered_joint_targets = [positions[name] for name in ordered_joint_names if name in positions]

        if ordered_joint_targets:
            scaled_joint_targets = [int(value * self.config.joint_scale) for value in ordered_joint_targets]
            self._send_command(
                {"command": self.config.move_command, "joint": scaled_joint_targets, "follow": False}
            )

        if self.config.has_gripper and "gripper" in positions:
            try:
                self._send_command(
                    {
                        "command": "set_gripper_position",
                        "position": int(positions["gripper"] * self.config.gripper_scale),
                        "block": False,
                    }
                )
            except Exception as exc:
                self._raise_required_gripper_error("set_gripper_position", exc)

        return positions

    def _read_joint_values(self) -> dict[str, float]:
        response = self._send_command_with_expected_key({"command": "get_joint_degree"}, expected_key="joint")
        raw_values = response.get("joint")
        if not isinstance(raw_values, list):
            raise RuntimeError(f"Unexpected RealMan joint response: {response}")

        ordered_joint_names = [name for name in self.config.joint_names if name != "gripper"]
        return {
            name: float(value) / self.config.joint_scale
            for name, value in zip(ordered_joint_names, raw_values, strict=False)
        }

    def _read_gripper_value(self) -> float:
        if self.config.gripper_feedback_source == "rm_plus":
            return self._read_rm_plus_gripper_position()
        if self.config.gripper_position_register is not None:
            return self._read_gripper_register_position()
        try:
            response = self._send_command_with_expected_key(
                {"command": "get_gripper_state"}, expected_key=self.config.gripper_state_position_key
            )
        except Exception as exc:
            self._raise_required_gripper_error("get_gripper_state", exc)
        self._last_gripper_value = self._get_gripper_state_position(response, "get_gripper_state")
        return self._last_gripper_value

    def _read_rm_plus_gripper_position(self) -> float:
        operation = "get_rm_plus_state_info"
        try:
            response = self._send_command_with_expected_key({"command": operation}, expected_key="state_info")
            state = response.get("state_info")
            if not isinstance(state, dict):
                raise RuntimeError(f"Expected RM+ state_info object: {response!r}")
            if "sys_state" in state and state["sys_state"] != 0:
                raise RuntimeError(f"RM+ feedback reports sys_state={state['sys_state']!r}: {response!r}")
            # Firmware uses sys_err; some protocol documentation calls it sys_error.
            if not any(name in state for name in ("sys_err", "sys_error")):
                raise RuntimeError(f"Missing RM+ system error field (sys_err/sys_error): {response!r}")
            for name in ("sys_err", "sys_error"):
                if name in state and state[name] != 0:
                    raise RuntimeError(f"RM+ feedback reports {name}={state[name]!r}: {response!r}")
            errors = state.get("dof_err")
            if not isinstance(errors, list) or len(errors) != 1 or errors[0] != 0:
                raise RuntimeError(f"Expected one error-free RM+ gripper DOF: {response!r}")
            positions = state.get("pos")
            if not isinstance(positions, list) or len(positions) != 1:
                raise RuntimeError(f"Expected one RM+ gripper position: {response!r}")
            raw_position = positions[0]
            if (
                isinstance(raw_position, bool)
                or not isinstance(raw_position, int | float)
                or not math.isfinite(raw_position)
            ):
                raise RuntimeError(f"RM+ gripper position must be a finite number: {response!r}")
            position = float(raw_position) / self.config.gripper_scale
            if not math.isfinite(position):
                raise RuntimeError(f"Scaled RM+ gripper position is not finite: {response!r}")
        except Exception as exc:
            self._raise_required_gripper_error(operation, exc)
        self._last_gripper_value = position
        return position

    def _read_gripper_register_position(self) -> float:
        address = self.config.gripper_position_register
        if address is None:
            raise RuntimeError("Gripper position register is not configured.")

        try:
            response = self._send_command_with_expected_key(
                {
                    "command": "read_multiple_holding_registers",
                    "port": self.config.gripper_modbus_port,
                    "address": address,
                    # RM65 requires at least three registers per batch. The first
                    # two contain the position; read and validate the third as well.
                    "num": 3,
                    "device": self.config.gripper_modbus_device,
                },
                expected_key="data",
            )
            data = response.get("data")
            if (
                not isinstance(data, list)
                or len(data) != 6
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
                    for value in data
                )
            ):
                raise RuntimeError(
                    f"Expected six big-endian register bytes for 0x{address:04X}-0x{address + 2:04X}: "
                    f"{response}"
                )
            raw_position = int.from_bytes(bytes(data[:4]), byteorder="big", signed=False)
            position = raw_position / self.config.gripper_scale
            if not math.isfinite(position):
                raise RuntimeError("Scaled gripper register position is not finite.")
        except Exception as exc:
            self._raise_required_gripper_error(
                f"read_multiple_holding_registers(0x{address:04X}-0x{address + 1:04X})", exc
            )

        self._last_gripper_value = position
        return self._last_gripper_value

    def _send_command_with_expected_key(
        self, payload: dict[str, Any], expected_key: str, max_attempts: int = 6
    ) -> dict[str, Any]:
        """
        Some controllers can emit asynchronous status notifications on the same socket.
        Send the request once, then consume socket messages until the expected
        response shape arrives.
        """
        self._send_payload(payload)
        response: dict[str, Any] = {}
        for _ in range(max_attempts):
            response = self._read_response()
            if "command" in response and response["command"] != payload.get("command"):
                continue
            if response.get("read_state") is False:
                raise RuntimeError(f"RealMan read_state=false for payload {payload}: {response}")
            if response.get("get_state") is False:
                raise RuntimeError(f"RealMan get_state=false for payload {payload}: {response}")
            if expected_key in response:
                return response
        raise RuntimeError(
            f"Unexpected RealMan response after {max_attempts} attempts for payload {payload}: {response}"
        )

    def _send_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._send_payload(payload)
        return self._read_response()

    def _send_payload(self, payload: dict[str, Any]) -> None:
        if self._socket is None or not self.is_connected:
            raise RuntimeError(f"RealMan arm is not connected: {asdict(self.config)}")

        message = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._socket.sendall(message.encode("utf-8"))

    def _read_response(self) -> dict[str, Any]:
        if self._socket is None or not self.is_connected:
            raise RuntimeError(f"RealMan arm is not connected: {asdict(self.config)}")

        while True:
            line = self._pop_buffered_line()
            if line:
                return json.loads(line)

            response = self._socket.recv(4096)
            decoded = response.decode("utf-8")
            if not decoded:
                return {}
            self._receive_buffer += decoded

    def _pop_buffered_line(self) -> str | None:
        while "\n" in self._receive_buffer:
            line, self._receive_buffer = self._receive_buffer.split("\n", 1)
            line = line.strip()
            if line:
                return line

        stripped = self._receive_buffer.strip()
        if not stripped:
            self._receive_buffer = ""
            return None

        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return None

        self._receive_buffer = ""
        return stripped


class RealManSDKClient(BaseRealManClient):
    """Optional adapter for the vendor `Robotic_Arm.rm_robot_interface` package."""

    def __init__(self, config: RealManClientConfig):
        super().__init__(config)
        self._sdk = importlib.import_module("Robotic_Arm.rm_robot_interface")
        self._robot: Any | None = None
        self._handle: Any | None = None

    def _require_connected(self) -> tuple[Any, Any]:
        if self._robot is None or self._handle is None or not self.is_connected:
            raise RuntimeError(f"RealMan arm is not connected: {asdict(self.config)}")
        return self._robot, self._handle

    def _raise_sdk_error(self, action: str, result: Any) -> RuntimeError:
        code = result[0] if isinstance(result, tuple) and result else result
        return RuntimeError(f"RealMan SDK call failed during `{action}` with result `{code}`.")

    def connect(self) -> None:
        thread_mode = self._sdk.rm_thread_mode_e.RM_TRIPLE_MODE_E
        robot_cls = self._sdk.RoboticArm
        self._robot = robot_cls(thread_mode)
        port = 8080 if self.config.port is None else self.config.port
        self._handle = self._robot.rm_create_robot_arm(self.config.host, port)

        handle_id = getattr(self._handle, "id", -1)
        if handle_id in (-1, None):
            raise RuntimeError(
                f"Failed to connect to RealMan arm at {self.config.host}:{port}. "
                f"Returned handle: {self._handle!r}"
            )

        self._is_connected = True
        self.software_info = self._query_software_info()

    def disconnect(self) -> None:
        if self._robot is not None and self._handle is not None:
            for method_name in ("rm_delete_robot_arm", "rm_destory", "rm_destroy"):
                method = getattr(self._robot, method_name, None)
                if callable(method):
                    try:
                        method()
                    except TypeError:
                        method(self._handle)
                    break

        self._handle = None
        self._robot = None
        self._is_connected = False

    def _query_software_info(self) -> dict[str, Any] | None:
        robot, _ = self._require_connected()
        query = getattr(robot, "rm_get_arm_software_info", None)
        if not callable(query):
            return None

        result = query()
        if not isinstance(result, tuple) or len(result) < 2:
            raise self._raise_sdk_error("rm_get_arm_software_info", result)
        if result[0] != 0:
            raise self._raise_sdk_error("rm_get_arm_software_info", result)
        return result[1]

    def get_joint_positions(self) -> dict[str, float]:
        robot, _ = self._require_connected()
        result = robot.rm_get_current_arm_joint()
        if not isinstance(result, tuple) or len(result) < 2 or result[0] != 0:
            raise self._raise_sdk_error("rm_get_current_arm_joint", result)
        payload = result[1]
        values = payload.get("joint") if isinstance(payload, dict) else payload
        if not isinstance(values, list | tuple):
            raise RuntimeError(f"Unexpected SDK joint payload: {payload}")
        ordered_joint_names = [name for name in self.config.joint_names if name != "gripper"]
        positions = dict(zip(ordered_joint_names, [float(value) for value in values], strict=False))
        if "gripper" in self.config.joint_names:
            positions["gripper"] = 0.0 if not self.config.has_gripper else self._get_gripper_position()
        return positions

    def send_joint_positions(self, positions: dict[str, float]) -> dict[str, float]:
        robot, _ = self._require_connected()
        ordered = [
            positions[name] for name in self.config.joint_names if name != "gripper" and name in positions
        ]
        if ordered:
            result = robot.rm_movej_canfd(ordered, False)
            if isinstance(result, tuple):
                ok = bool(result) and result[0] == 0
            else:
                ok = result == 0 or result is None
            if not ok:
                raise self._raise_sdk_error("rm_movej_canfd", result)
        if self.config.has_gripper and "gripper" in positions:
            self._set_gripper_position(positions["gripper"])
        return positions

    def _get_gripper_position(self) -> float:
        robot, _ = self._require_connected()
        query = getattr(robot, "rm_get_gripper_state", None)
        if not callable(query):
            self._raise_required_gripper_error("rm_get_gripper_state")

        result = query()
        if not isinstance(result, tuple) or len(result) < 2 or result[0] != 0:
            self._raise_required_gripper_error(
                "rm_get_gripper_state", self._raise_sdk_error("rm_get_gripper_state", result)
            )
        payload = result[1]
        return self._get_gripper_state_position(payload, "rm_get_gripper_state")

    def _set_gripper_position(self, position: float) -> None:
        robot, _ = self._require_connected()
        target = int(position * self.config.gripper_scale)
        for method_name in ("rm_set_gripper_position", "rm_set_gripper_pos"):
            method = getattr(robot, method_name, None)
            if callable(method):
                result = method(target, self.config.gripper_speed, False)
                if isinstance(result, tuple):
                    ok = bool(result) and result[0] == 0
                else:
                    ok = result == 0 or result is None
                if not ok:
                    self._raise_required_gripper_error(
                        method_name, self._raise_sdk_error(method_name, result)
                    )
                return
        self._raise_required_gripper_error("rm_set_gripper_position")


class RealManAlohaSDKClient(BaseRealManClient):
    """Adapter for the cleaned RM_Aloha SDK wrapper bundled with LeRobot."""

    def __init__(self, config: RealManClientConfig):
        super().__init__(config)
        self._sdk = importlib.import_module("lerobot.common.realman.rm_aloha_sdk")
        self._arm: Any | None = None

    def _require_connected(self) -> Any:
        if self._arm is None or not self.is_connected:
            raise RuntimeError(f"RealMan arm is not connected: {asdict(self.config)}")
        return self._arm

    def _raise_sdk_error(self, action: str, result: Any) -> RuntimeError:
        return RuntimeError(f"RealMan Aloha SDK call failed during `{action}` with result `{result}`.")

    def _is_ok(self, result: Any) -> bool:
        return result == 0 or result is None

    def connect(self) -> None:
        arm_cls = self._sdk.Arm
        port = 8080 if self.config.port is None else self.config.port
        self._arm = arm_cls(self.config.arm_model, self.config.host, port)
        state = self._arm.Arm_Socket_State()
        if not self._is_ok(state):
            raise self._raise_sdk_error("Arm_Socket_State", state)
        self._is_connected = True

    def disconnect(self) -> None:
        if self._arm is not None:
            close = getattr(self._arm, "Arm_Socket_Close", None)
            if callable(close):
                close()
            uninit = getattr(self._arm, "RM_API_UnInit", None)
            if callable(uninit):
                uninit()
        self._arm = None
        self._is_connected = False

    def configure(self) -> None:
        if self.config.has_gripper and self.config.initialize_gripper_route:
            arm = self._require_connected()
            result = arm.Set_Gripper_Route(0, self.config.gripper_open_value, False)
            if not self._is_ok(result):
                self._raise_required_gripper_error(
                    "Set_Gripper_Route", self._raise_sdk_error("Set_Gripper_Route", result)
                )

    def get_joint_positions(self) -> dict[str, float]:
        arm = self._require_connected()
        result, values = arm.Get_Joint_Degree()
        if not self._is_ok(result):
            raise self._raise_sdk_error("Get_Joint_Degree", result)
        if not isinstance(values, list | tuple):
            raise RuntimeError(f"Unexpected RealMan Aloha SDK joint payload: {values}")

        ordered_joint_names = [name for name in self.config.joint_names if name != "gripper"]
        positions = {name: float(value) for name, value in zip(ordered_joint_names, values, strict=False)}
        if "gripper" in self.config.joint_names:
            positions["gripper"] = 0.0 if not self.config.has_gripper else self._get_gripper_position()
        return positions

    def send_joint_positions(self, positions: dict[str, float]) -> dict[str, float]:
        arm = self._require_connected()
        ordered = [
            positions[name] for name in self.config.joint_names if name != "gripper" and name in positions
        ]
        if ordered:
            result = arm.Movej_CANFD(ordered, False)
            if not self._is_ok(result):
                raise self._raise_sdk_error("Movej_CANFD", result)

        if self.config.has_gripper and "gripper" in positions:
            target = int(positions["gripper"] * self.config.gripper_scale)
            result = arm.Set_Gripper_Position(target, False)
            if not self._is_ok(result):
                self._raise_required_gripper_error(
                    "Set_Gripper_Position", self._raise_sdk_error("Set_Gripper_Position", result)
                )
        return positions

    def _get_gripper_position(self) -> float:
        arm = self._require_connected()
        result, state = arm.Get_Gripper_State()
        if not self._is_ok(result):
            self._raise_required_gripper_error(
                "Get_Gripper_State", self._raise_sdk_error("Get_Gripper_State", result)
            )
        return self._get_gripper_state_position(state, "Get_Gripper_State")


def make_realman_client(config: RealManClientConfig) -> BaseRealManClient:
    backend = config.sdk_backend or "socket_json"
    if backend == "socket_json":
        return RealManSocketClient(config)
    if backend == "rm_aloha_sdk":
        try:
            return RealManAlohaSDKClient(config)
        except ImportError:
            return PlaceholderRealManClient(config)
    if backend == "vendor_sdk":
        try:
            return RealManSDKClient(config)
        except ImportError:
            return PlaceholderRealManClient(config)
    return PlaceholderRealManClient(config)

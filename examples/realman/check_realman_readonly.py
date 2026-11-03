#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Read RealMan feedback without motion, initialization or device configuration writes.

TCP commands are restricted to status queries. Master serial ports receive only
the fixed state-query bytes. Configured wrist cameras provide one RGB frame;
images are not saved. Direct Modbus reads require --probe-modbus because they
can time out when the gripper link is managed by RM+. This script never creates
or appends a dataset.
"""

from __future__ import annotations

import argparse
import json
import socket
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

READ_COMMANDS = frozenset(
    {
        "get_joint_degree",
        "get_gripper_state",
        "get_arm_software_info",
        "get_tool_voltage",
        "get_tool_RS485_mode",
        "get_controller_RS485_mode",
        "get_rm_plus_mode",
        "get_rm_plus_base_info",
        "get_rm_plus_state_info",
        "read_holding_registers",
        "read_multiple_holding_registers",
    }
)
MASTER_QUERY = bytes.fromhex("55AA02000067")
RM_PLUS_BAUDRATES = frozenset({9600, 115200, 256000, 460800})


def matches_response(command: str, response: Any) -> bool:
    """Match explicit protocol replies, including the older RM65 aliases."""
    if not isinstance(response, dict):
        return False
    reply_command = response.get("command")
    if reply_command == command:
        return True
    if command == "get_arm_software_info" and reply_command == "arm_software_info":
        return True
    if command == "get_tool_voltage" and reply_command is None:
        return response.get("state") == "tool_voltage_state" and "voltage_type" in response
    return (
        command == "get_joint_degree"
        and reply_command is None
        and response.get("state") == "joint_degree"
        and "joint" in response
    )


def error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


class ReadOnlyJsonClient:
    """A bounded, command-correlated transport with no configuration API."""

    def __init__(self, host: str, port: int, timeout_s: float):
        self.timeout_s = timeout_s
        self.buffer = b""
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.settimeout(timeout_s)

    def close(self) -> None:
        self.sock.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("command") not in READ_COMMANDS:
            raise ValueError(f"Refusing non-read-only command: {payload.get('command')!r}")
        started = time.perf_counter()
        result: dict[str, Any] = {"payload": payload, "raw_replies": [], "replies": []}
        try:
            self.sock.sendall((json.dumps(payload, separators=(",", ":")) + "\r\n").encode())
            deadline = started + self.timeout_s
            for _ in range(64):
                while b"\n" not in self.buffer:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("No matching command response before deadline")
                    self.sock.settimeout(remaining)
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("Controller closed the connection")
                    self.buffer += chunk
                    if len(self.buffer) > 65536:
                        raise RuntimeError("Controller response exceeds 64 KiB")
                line, self.buffer = self.buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                result["raw_replies"].append(line.decode("utf-8", errors="replace"))
                response = json.loads(line)
                result["replies"].append(response)
                if matches_response(payload["command"], response):
                    result["response"] = response
                    if response.get("read_state") is False:
                        result["error"] = "Controller returned read_state=false"
                    if response.get("get_state") is False:
                        result["error"] = "Controller returned get_state=false"
                    break
            else:
                raise RuntimeError("No matching command in 64 response lines")
        except Exception as exc:
            result["error"] = error_text(exc)
            result["transport_failed"] = isinstance(exc, OSError | ConnectionError)
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return result


def read_follower(
    config: dict[str, Any], samples: int, timeout_s: float, probe_modbus: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": config.get("host"),
        "port": config.get("port", 8080),
        "queries": [],
        "note": "Responses are evidence, not proof that feedback is valid or calibrated.",
        "native_gripper": {
            "note": "Native get_gripper_state feedback may not represent a CTAG2F connected through RM+.",
            "samples": [],
        },
        "rm_plus": {
            "status": "not_queried",
            "enabled": None,
            "mode": None,
            "samples": [],
            "errors": [],
            "note": "RM+ state is reported separately from native gripper state; this check never enables RM+.",
        },
        "modbus_probe": {
            "enabled": probe_modbus,
            "queries": [],
            "note": "Direct register queries are opt-in and may time out while RM+ manages the gripper link.",
        },
    }
    started = time.perf_counter()
    try:
        client = ReadOnlyJsonClient(config["host"], config.get("port") or 8080, timeout_s)
    except Exception as exc:
        result["connection_error"] = error_text(exc)
        result["connect_elapsed_ms"] = (time.perf_counter() - started) * 1000
        return result
    result["connect_elapsed_ms"] = (time.perf_counter() - started) * 1000
    try:

        def query_status(payload: dict[str, Any], index: int = -1) -> dict[str, Any]:
            query = client.request(payload)
            query["sample"] = index
            response = query.get("response", {})
            command = payload["command"]
            if command == "get_tool_voltage":
                query["interpretation"] = (
                    "Controller output setting only; this does not measure voltage at the gripper "
                    "or verify its wiring."
                )
            elif command == "get_gripper_state":
                query["source"] = "native_gripper"
                query["interpretation"] = result["native_gripper"]["note"]
                result["native_gripper"]["samples"].append(query)
            elif command.startswith("get_rm_plus_"):
                query["source"] = "rm_plus"
                field = {
                    "get_rm_plus_base_info": "base_info",
                    "get_rm_plus_state_info": "state_info",
                }.get(command)
                if field and not isinstance(response.get(field), dict) and "error" not in query:
                    query["error"] = f"RM+ response does not contain a {field} object"
                if command == "get_rm_plus_state_info":
                    result["rm_plus"]["samples"].append(query)
                if "error" in query:
                    result["rm_plus"]["status"] = "error"
                    result["rm_plus"]["errors"].append(
                        {"command": command, "sample": index, "error": query["error"]}
                    )
            elif command.startswith("read_"):
                query["source"] = "direct_modbus"
                result["modbus_probe"]["queries"].append(query)
            data = response.get("data")
            if command == "read_multiple_holding_registers" and isinstance(data, list):
                if len(data) == 6 and all(type(value) is int and 0 <= value <= 255 for value in data):
                    query["raw_u32_big_endian"] = int.from_bytes(bytes(data[:4]), "big")
                else:
                    query["decode_error"] = "Expected six big-endian bytes; raw response preserved"
            result["queries"].append(query)
            if query.get("transport_failed"):
                result["aborted"] = "Transport failed; no reconnection or remaining requests attempted"
            return query

        for command in (
            "get_arm_software_info",
            "get_tool_voltage",
            "get_tool_RS485_mode",
            "get_controller_RS485_mode",
            "get_rm_plus_mode",
        ):
            query = query_status({"command": command})
            if query.get("transport_failed"):
                return result

        rm_plus = result["rm_plus"]
        rm_plus["mode_query"] = query
        mode = query.get("response", {}).get("mode")
        rm_plus["mode"] = mode
        if "error" in query:
            rm_plus["status"] = "error"
        elif type(mode) is not int or mode < 0:
            rm_plus["status"] = "error"
            rm_plus["errors"].append({"command": "get_rm_plus_mode", "error": f"Invalid RM+ mode: {mode!r}"})
        elif mode == 0:
            rm_plus["status"] = "disabled"
            rm_plus["enabled"] = False
        else:
            rm_plus["status"] = "enabled"
            rm_plus["enabled"] = True
            rm_plus["baudrate"] = mode
            if mode not in RM_PLUS_BAUDRATES:
                rm_plus["mode_note"] = (
                    f"Unexpected RM+ baudrate; documented values are {sorted(RM_PLUS_BAUDRATES)}"
                )
            base_query = query_status({"command": "get_rm_plus_base_info"})
            rm_plus["base_info"] = base_query.get("response", {}).get("base_info")
            if base_query.get("transport_failed"):
                return result

        base = {
            "port": config.get("gripper_modbus_port", 1),
            "device": config.get("gripper_modbus_device", 1),
        }
        for index in range(samples):
            payloads = [{"command": "get_joint_degree"}, {"command": "get_gripper_state"}]
            if rm_plus["enabled"]:
                payloads.append({"command": "get_rm_plus_state_info"})
            if probe_modbus:
                payloads.extend(
                    [
                        {"command": "read_multiple_holding_registers", **base, "address": 0x0609, "num": 3},
                        {"command": "read_multiple_holding_registers", **base, "address": 0x060D, "num": 3},
                        {"command": "read_holding_registers", **base, "address": 0x0604},
                        {"command": "read_holding_registers", **base, "address": 0x0612},
                    ]
                )
            for payload in payloads:
                query = query_status(payload, index)
                if query.get("transport_failed"):
                    return result
    except Exception as exc:
        result["error"] = error_text(exc)
    finally:
        try:
            client.close()
        except Exception as exc:
            result["close_error"] = error_text(exc)
    return result


def serial_inventory() -> dict[str, Any]:
    return {
        "ttyUSB": [str(path) for path in sorted(Path("/dev").glob("ttyUSB*"))],
        "by_id": {str(path): str(path.resolve()) for path in sorted(Path("/dev/serial/by-id").glob("*"))},
    }


def read_master(config: dict[str, Any], samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "port": config.get("port"),
        "exists": Path(config["port"]).exists(),
        "query_hex": MASTER_QUERY.hex().upper(),
        "samples": [],
    }
    master = None
    try:
        configured_query = bytes.fromhex(config.get("init_hex_command", MASTER_QUERY.hex()))
        if configured_query != MASTER_QUERY:
            raise ValueError("Refusing custom master init/query command; only 55AA02000067 is permitted")
        from lerobot.teleoperators.rm_aloha_master import RMAlohaMaster, RMAlohaMasterConfig

        with tempfile.TemporaryDirectory(prefix="realman-readonly-") as calibration_dir:
            master_config = {**config, "calibration_dir": Path(calibration_dir)}
            # A nonblocking read keeps frame_timeout_s effective even if a YAML
            # file requested an unbounded serial timeout.
            master_config["timeout_s"] = 0.0
            master = RMAlohaMaster(RMAlohaMasterConfig(**master_config))
            master.serial.open()
            for index in range(samples):
                started = time.perf_counter()
                sample: dict[str, Any] = {"sample": index, "query_hex": MASTER_QUERY.hex().upper()}
                try:
                    master.serial.write(MASTER_QUERY)
                    encoded = master._read_encoded_frame()
                    sample["response_hex"] = encoded
                    sample["state"] = dict(
                        zip(master.action_features, master._decode_master_state(encoded), strict=True)
                    )
                except Exception as exc:
                    sample["error"] = error_text(exc)
                sample["elapsed_ms"] = (time.perf_counter() - started) * 1000
                result["samples"].append(sample)
                if "error" in sample:
                    break
    except Exception as exc:
        result["error"] = error_text(exc)
    finally:
        if master is not None:
            master.serial.close()
    return result


def read_cameras(robot: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    result: dict[str, Any] = {"enumerated": [], "configured": {}}
    try:
        import pyrealsense2 as rs

        context = rs.context()
        for device in context.query_devices():
            result["enumerated"].append(
                {
                    "serial": device.get_info(rs.camera_info.serial_number),
                    "name": device.get_info(rs.camera_info.name),
                }
            )
        for side in ("left", "right"):
            camera = robot[f"{side}_arm_config"].get("cameras", {}).get("wrist")
            if camera is None:
                continue
            item: dict[str, Any] = {"config": camera}
            result["configured"][f"{side}_wrist"] = item
            pipeline = None
            started = False
            begin = time.perf_counter()
            try:
                if camera.get("type") != "intelrealsense":
                    raise ValueError("Only intelrealsense wrist cameras are supported")
                serial_or_name = str(camera["serial_number_or_name"])
                matches = [
                    device["serial"]
                    for device in result["enumerated"]
                    if serial_or_name in (device["serial"], device["name"])
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one matching camera for {serial_or_name!r}, found {matches}"
                    )
                stream_config = rs.config()
                stream_config.enable_device(matches[0])
                stream_config.enable_stream(
                    rs.stream.color,
                    camera.get("width", 640),
                    camera.get("height", 480),
                    rs.format.rgb8,
                    camera.get("fps", 30),
                )
                pipeline = rs.pipeline(context)
                pipeline.start(stream_config)
                started = True
                frame = pipeline.wait_for_frames(int(timeout_s * 1000)).get_color_frame()
                if not frame:
                    raise RuntimeError("No color frame")
                item["frame"] = {
                    "width": frame.get_width(),
                    "height": frame.get_height(),
                    "frame_number": frame.get_frame_number(),
                    "timestamp_ms": frame.get_timestamp(),
                }
            except Exception as exc:
                item["error"] = error_text(exc)
            finally:
                if pipeline is not None and started:
                    try:
                        pipeline.stop()
                    except Exception as exc:
                        item["stop_error"] = error_text(exc)
                item["elapsed_ms"] = (time.perf_counter() - begin) * 1000
    except Exception as exc:
        result["error"] = error_text(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/hardware/realman_double_arm_hardware_config.local.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--skip-followers", action="store_true")
    parser.add_argument("--skip-masters", action="store_true")
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument(
        "--probe-modbus",
        action="store_true",
        help="Also read CTAG2F registers directly; RM+ may own this link, causing these optional queries to time out.",
    )
    args = parser.parse_args()
    if args.samples < 1 or not 0 < args.timeout_s <= 30:
        parser.error("samples must be positive; timeout-s must be in (0, 30]")
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": str(args.config_path.resolve()),
        "serial_inventory": serial_inventory(),
        "followers": {},
        "masters": {},
        "cameras": {},
        "skipped": [name for name in ("followers", "masters", "cameras") if getattr(args, f"skip_{name}")],
    }
    try:
        config = yaml.safe_load(args.config_path.read_text())
        for side in ("left", "right"):
            if not args.skip_followers:
                result["followers"][side] = read_follower(
                    config["robot"][f"{side}_arm_config"],
                    args.samples,
                    args.timeout_s,
                    probe_modbus=args.probe_modbus,
                )
            if not args.skip_masters:
                result["masters"][side] = read_master(config["teleop"][f"{side}_arm_config"], args.samples)
        if not args.skip_cameras:
            result["cameras"] = read_cameras(config["robot"], args.timeout_s)
    except Exception as exc:
        result["error"] = error_text(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Read-only check saved to {args.output}; inspect errors and raw responses before drawing conclusions."
    )


if __name__ == "__main__":
    main()

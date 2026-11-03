#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Watch both RM+ grippers using only status queries and the production feedback parser.

No motion, initialization, mode changes, cameras, serial ports or datasets.
Observed changes do not by themselves prove agreement with physical movement.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from lerobot.common.realman.client import RealManClientConfig, RealManSocketClient

READ_COMMANDS = frozenset({"get_rm_plus_mode", "get_rm_plus_base_info", "get_rm_plus_state_info"})
SIDES = ("left", "right")


def _json_safe(value: Any) -> Any:
    """Keep invalid nonfinite raw values visible as text in strict JSON logs."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class ReadOnlyGripperClient(RealManSocketClient):
    """Keep production transport and validation, with a final payload whitelist."""

    def __init__(self, config: RealManClientConfig):
        super().__init__(config)
        self.last_response: dict[str, Any] | None = None
        self._command: str | None = None

    def _send_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("command") not in READ_COMMANDS:
            raise ValueError(f"Refusing non-monitor command: {payload.get('command')!r}")
        self._command = payload["command"]
        self.last_response = None
        super()._send_payload(payload)

    def _read_response(self) -> dict[str, Any]:
        response = super()._read_response()
        if response.get("command") in (None, self._command):
            self.last_response = response
        return response

    def preflight(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode_enabled": None, "device_online": False}
        stage = "connect"
        try:
            self.connect()
            stage = "mode"
            response = self._send_command_with_expected_key({"command": "get_rm_plus_mode"}, "mode")
            result["mode_response"] = response
            mode = response["mode"]
            if type(mode) is not int or mode <= 0:
                result["mode_enabled"] = False if mode == 0 else None
                raise RuntimeError(
                    f"RM+ is disabled or mode is invalid: {mode!r}; this tool will not enable it"
                )
            result["mode_enabled"] = True
            stage = "base"
            response = self._send_command_with_expected_key({"command": "get_rm_plus_base_info"}, "base_info")
            result["base_response"] = response
            base = response["base_info"]
            if not isinstance(base, dict) or type(base.get("dof")) is not int or base["dof"] != 1:
                raise RuntimeError("RM+ metadata must describe exactly one gripper DOF")
            if base.get("pos_low") != [0] or base.get("pos_up") != [self.config.gripper_scale]:
                raise RuntimeError(
                    f"RM+ position range {base.get('pos_low')}..{base.get('pos_up')} does not match "
                    f"the configured normalization range 0..{self.config.gripper_scale}"
                )
            result["device_online"] = True
        except Exception as exc:
            if stage in ("mode", "base") and self.last_response is not None:
                result[f"{stage}_response"] = self.last_response
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def read_sample(self) -> dict[str, Any]:
        self.last_response = None
        result: dict[str, Any] = {"raw_position": None, "normalized": None}
        try:
            # This is the same parser used by RealManFollower.get_observation().
            normalized = self._read_gripper_value()
            state = self.last_response["state_info"]
            raw_position = state["pos"][0]
            if not 0 <= raw_position <= self.config.gripper_scale:
                raise RuntimeError(f"RM+ position {raw_position!r} is outside the metadata range")
            result.update(raw_position=raw_position, normalized=normalized)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["raw_state"] = self.last_response
        return result


def client_configs(config: dict[str, Any]) -> dict[str, RealManClientConfig]:
    robot = config.get("robot", {})
    if robot.get("type") != "double_realman_follower":
        raise ValueError("Expected robot.type=double_realman_follower")
    result = {}
    names = {item.name for item in fields(RealManClientConfig)}
    for side in SIDES:
        path = f"robot.{side}_arm_config"
        arm = robot[f"{side}_arm_config"]
        if arm.get("gripper_feedback_source") != "rm_plus":
            raise ValueError(f"{path}.gripper_feedback_source must be rm_plus")
        for option, default in (("initialize_gripper_route", True), ("initialize_gripper_modbus", False)):
            if arm.get(option, default) is not False:
                raise ValueError(f"{path}.{option} must be false; no devices were connected")
        result[side] = RealManClientConfig(**{key: value for key, value in arm.items() if key in names})
        if not result[side].has_gripper:
            raise ValueError(f"{path}.has_gripper must be true")
        for option in ("connect_timeout_s", "read_timeout_s"):
            value = getattr(result[side], option)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{path}.{option} must be a positive finite timeout")
    return result


def watch_grippers(
    config: dict[str, Any], duration_s: float = 30, fps: float = 10, output: Path | None = None
) -> dict[str, Any]:
    if not math.isfinite(duration_s) or duration_s <= 0 or not math.isfinite(fps) or not 1 <= fps <= 30:
        raise ValueError("duration-s must be positive and finite; fps must be in [1, 30]")
    configs = client_configs(config)
    summary: dict[str, Any] = {
        "event": "summary",
        "preflight_ok": False,
        "interrupted": False,
        "sides": {
            side: {"attempts": 0, "sample_count": 0, "min": None, "max": None, "changed": False, "errors": 0}
            for side in SIDES
        },
        "note": "changed only means a change was observed; it does not prove physical agreement or a successful fix.",
    }
    first_positions = {}
    clients = {}
    with ExitStack() as stack:
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
        writer = stack.enter_context(output.open("x", encoding="utf-8")) if output is not None else None

        def emit(record):
            if writer is not None:
                writer.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
                writer.flush()

        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="realman_gripper_watch")
        try:
            clients = {side: ReadOnlyGripperClient(configs[side]) for side in SIDES}
            futures = {side: executor.submit(clients[side].preflight) for side in SIDES}
            wait(futures.values())
            preflights = {side: futures[side].result() for side in SIDES}
            for side, preflight in preflights.items():
                emit({"event": "preflight", "side": side, "host": configs[side].host, **preflight})
                if "error" in preflight:
                    summary["sides"][side]["errors"] += 1
                    print(
                        f"{side}: PREFLIGHT ERROR {preflight['error']} (enabled does not mean device online)"
                    )
                else:
                    print(f"{side}: RM+ preflight passed; mode={preflight['mode_response']['mode']}")
            summary["preflight_ok"] = all("error" not in item for item in preflights.values())
            if summary["preflight_ok"]:
                started = time.monotonic()
                while time.monotonic() - started < duration_s:
                    tick = time.monotonic()
                    futures = {side: executor.submit(clients[side].read_sample) for side in SIDES}
                    wait(futures.values())
                    for side in SIDES:
                        sample = futures[side].result()
                        elapsed = time.monotonic() - started
                        emit({"event": "sample", "side": side, "elapsed_s": elapsed, **sample})
                        stats = summary["sides"][side]
                        stats["attempts"] += 1
                        if "error" in sample:
                            stats["errors"] += 1
                            print(f"{elapsed:7.3f}s {side}: ERROR {sample['error']}")
                            continue
                        value = sample["normalized"]
                        stats["sample_count"] += 1
                        stats["min"] = value if stats["min"] is None else min(stats["min"], value)
                        stats["max"] = value if stats["max"] is None else max(stats["max"], value)
                        first_positions.setdefault(side, sample["raw_position"])
                        stats["changed"] |= sample["raw_position"] != first_positions[side]
                        print(f"{elapsed:7.3f}s {side}: raw={sample['raw_position']} normalized={value:.6f}")
                    remaining = min(
                        1 / fps - (time.monotonic() - tick), duration_s - (time.monotonic() - started)
                    )
                    if remaining > 0:
                        time.sleep(remaining)
            else:
                summary["stopped_reason"] = "preflight_failed; sampling was not started"
        except KeyboardInterrupt:
            summary["interrupted"] = True
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            for side, client in clients.items():
                try:
                    client.disconnect()
                except Exception as exc:
                    summary["sides"][side]["errors"] += 1
                    summary["sides"][side]["close_error"] = f"{type(exc).__name__}: {exc}"
            emit(summary)
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/hardware/realman_double_arm_hardware_config.local.yaml"),
    )
    parser.add_argument("--duration-s", type=float, default=30)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument(
        "--output", type=Path, help="Optional new JSONL path; existing files are never overwritten"
    )
    args = parser.parse_args()
    try:
        summary = watch_grippers(
            yaml.safe_load(args.config_path.read_text()), args.duration_s, args.fps, args.output
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if summary["interrupted"]:
        raise SystemExit(130)
    if not summary["preflight_ok"] or any(item["errors"] for item in summary["sides"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

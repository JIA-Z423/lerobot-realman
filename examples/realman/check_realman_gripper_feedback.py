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

"""Compare RealMan gripper control and feedback through socket JSON and SDK."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if REPO_SRC.exists():
    sys.path.insert(0, str(REPO_SRC))


def _request_json(host: str, port: int, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall((json.dumps(payload, separators=(",", ":")) + "\r\n").encode())
        buffer = ""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError(f"empty response for {payload}")
            buffer += chunk.decode("utf-8", errors="replace")
            for raw_line in buffer.splitlines():
                line = raw_line.strip()
                if line:
                    return json.loads(line)


def _print_payload(label: str, payload: Any) -> None:
    print(f"    {label}: {payload}")


def _socket_check(args: argparse.Namespace, host: str) -> None:
    print(f"\n[{host} socket_json]")
    _print_payload(
        "set_gripper_route",
        _request_json(
            host,
            args.port,
            {"command": "set_gripper_route", "min": args.route_min, "max": args.route_max},
            args.timeout_s,
        ),
    )
    _print_payload(
        "initial get_gripper_state",
        _request_json(host, args.port, {"command": "get_gripper_state"}, args.timeout_s),
    )

    for position in args.positions:
        _print_payload(
            f"set_gripper_position({position})",
            _request_json(
                host,
                args.port,
                {"command": "set_gripper_position", "position": position, "block": False},
                args.timeout_s,
            ),
        )
        time.sleep(args.settle_s)
        for sample_idx in range(args.samples):
            _print_payload(
                f"get_gripper_state[{sample_idx}]",
                _request_json(host, args.port, {"command": "get_gripper_state"}, args.timeout_s),
            )
            if sample_idx < args.samples - 1:
                time.sleep(args.interval_s)


def _sdk_state_to_dict(state: Any) -> dict[str, Any]:
    return {
        "enable_state": bool(getattr(state, "enable_state")),
        "status": getattr(state, "status"),
        "error": getattr(state, "error"),
        "mode": getattr(state, "mode"),
        "current_force": getattr(state, "current_force"),
        "temperature": getattr(state, "temperature"),
        "actpos": getattr(state, "actpos"),
    }


def _sdk_check(args: argparse.Namespace, host: str) -> None:
    print(f"\n[{host} rm_aloha_sdk]")
    from lerobot.common.realman.rm_aloha_sdk import Arm

    arm = Arm(args.arm_model, host, args.port)
    try:
        _print_payload("Arm_Socket_State", arm.Arm_Socket_State())
        _print_payload("Set_Gripper_Route", arm.Set_Gripper_Route(args.route_min, args.route_max, False))
        result, state = arm.Get_Gripper_State()
        _print_payload("initial Get_Gripper_State", {"result": result, **_sdk_state_to_dict(state)})

        for position in args.positions:
            _print_payload(
                f"Set_Gripper_Position({position})",
                arm.Set_Gripper_Position(position, False, args.sdk_timeout_s),
            )
            time.sleep(args.settle_s)
            for sample_idx in range(args.samples):
                result, state = arm.Get_Gripper_State()
                _print_payload(
                    f"Get_Gripper_State[{sample_idx}]",
                    {"result": result, **_sdk_state_to_dict(state)},
                )
                if sample_idx < args.samples - 1:
                    time.sleep(args.interval_s)
    finally:
        arm.Arm_Socket_Close()
        arm.RM_API_UnInit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=["192.168.1.19", "192.168.1.20"])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--arm-model", type=int, default=65)
    parser.add_argument("--backend", choices=("socket", "sdk", "both"), default="both")
    parser.add_argument("--route-min", type=int, default=0)
    parser.add_argument("--route-max", type=int, default=1000)
    parser.add_argument("--positions", nargs="+", type=int, default=[1000, 0])
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--sdk-timeout-s", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for host in args.hosts:
        if args.backend in ("socket", "both"):
            _socket_check(args, host)
        if args.backend in ("sdk", "both"):
            _sdk_check(args, host)


if __name__ == "__main__":
    main()

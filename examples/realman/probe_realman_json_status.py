#!/usr/bin/env python

"""Read-only probe for RealMan socket JSON status commands."""

from __future__ import annotations

import argparse
import json
import socket
from typing import Any


READ_ONLY_COMMANDS = [
    {"command": "get_joint_degree"},
    {"command": "get_current_joint_current"},
    {"command": "get_gripper_state"},
    {"command": "get_arm_state"},
    {"command": "get_current_arm_state"},
    {"command": "get_joint_state"},
    {"command": "get_controller_state"},
]


def request(host: str, port: int, payload: dict[str, Any], timeout_s: float) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall((json.dumps(payload, separators=(",", ":")) + "\r\n").encode())
            data = sock.recv(4096).decode(errors="replace").strip()
            return data or "<empty>"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=["192.168.1.19", "192.168.1.20"])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for host in args.hosts:
        print(f"\n== {host} read-only status probes ==")
        for payload in READ_ONLY_COMMANDS:
            print("SEND", payload)
            print("RECV", request(host, args.port, payload, args.timeout_s))


if __name__ == "__main__":
    main()

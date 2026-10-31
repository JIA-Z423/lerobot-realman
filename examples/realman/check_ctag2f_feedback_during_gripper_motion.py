#!/usr/bin/env python

"""Move only the CTAG2F gripper and sample Modbus feedback registers."""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any


REGISTERS = [
    ("target_high", 0x0102),
    ("target_low", 0x0103),
    ("torque_reached", 0x0601),
    ("position_reached", 0x0602),
    ("ready", 0x0604),
    ("pos_high", 0x0609),
    ("pos_low", 0x060A),
    ("user_pos_high", 0x060D),
    ("user_pos_low", 0x060E),
    ("speed", 0x060B),
    ("current", 0x060C),
    ("alarm", 0x0612),
]


class JsonSocket:
    def __init__(self, host: str, port: int, timeout_s: float):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.settimeout(timeout_s)
        self.buffer = ""

    def close(self) -> None:
        self.sock.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any] | str:
        self.sock.sendall((json.dumps(payload, separators=(",", ":")) + "\r\n").encode())
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                return "<empty>"
            self.buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in self.buffer:
                raw_line, self.buffer = self.buffer.split("\n", 1)
                line = raw_line.strip()
                if line:
                    return json.loads(line)


def read_register(client: JsonSocket, args: argparse.Namespace, address: int) -> int | None:
    response = client.request(
        {
            "command": "read_holding_registers",
            "port": args.modbus_port,
            "address": address,
            "device": args.device,
        }
    )
    if isinstance(response, dict) and isinstance(response.get("data"), int | float):
        return int(response["data"])
    return None


def sample(client: JsonSocket, args: argparse.Namespace) -> dict[str, int | None]:
    values = {name: read_register(client, args, address) for name, address in REGISTERS}
    target_high = values.get("target_high")
    target_low = values.get("target_low")
    pos_high = values.get("pos_high")
    pos_low = values.get("pos_low")
    user_pos_high = values.get("user_pos_high")
    user_pos_low = values.get("user_pos_low")
    values["target"] = None if target_high is None or target_low is None else (target_high << 16) + target_low
    values["pos"] = None if pos_high is None or pos_low is None else (pos_high << 16) + (pos_low & 0xFFFF)
    values["user_pos"] = (
        None
        if user_pos_high is None or user_pos_low is None
        else (user_pos_high << 16) + (user_pos_low & 0xFFFF)
    )
    return values


def position_to_bytes(position: int) -> list[int]:
    high = (position >> 16) & 0xFFFF
    low = position & 0xFFFF
    return [(high >> 8) & 0xFF, high & 0xFF, (low >> 8) & 0xFF, low & 0xFF]


def command_gripper(client: JsonSocket, args: argparse.Namespace, position: int) -> list[dict[str, Any] | str]:
    if args.motion_command == "set_gripper_position":
        return [
            client.request({"command": "set_gripper_position", "position": position, "block": False})
        ]
    return [
        client.request(
            {
                "command": "write_registers",
                "port": args.modbus_port,
                "address": 0x0102,
                "num": 2,
                "data": position_to_bytes(position),
                "device": args.device,
            }
        ),
        client.request(
            {
                "command": "write_single_register",
                "port": args.modbus_port,
                "address": 0x0108,
                "data": 1,
                "device": args.device,
            }
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=["192.168.1.19", "192.168.1.20"])
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--modbus-port", type=int, default=1)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--modbus-timeout", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--positions", nargs="+", type=int, default=[1000, 0])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument(
        "--motion-command",
        choices=("set_gripper_position", "modbus"),
        default="set_gripper_position",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for host in args.hosts:
        print(f"\n== {host} CTAG2F gripper-motion feedback ==")
        client = JsonSocket(host, args.arm_port, args.timeout_s)
        try:
            print(
                "set_modbus_mode",
                client.request(
                    {
                        "command": "set_modbus_mode",
                        "port": args.modbus_port,
                        "baudrate": args.baudrate,
                        "timeout": args.modbus_timeout,
                    }
                ),
            )
            print("initial", sample(client, args))
            for position in args.positions:
                print(f"{args.motion_command}({position})", command_gripper(client, args, position))
                for sample_idx in range(args.samples):
                    print(f"sample[{position}][{sample_idx}]", sample(client, args))
                    if sample_idx < args.samples - 1:
                        time.sleep(args.interval_s)
        finally:
            client.close()


if __name__ == "__main__":
    main()

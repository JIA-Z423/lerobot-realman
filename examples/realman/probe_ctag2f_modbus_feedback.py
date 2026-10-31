#!/usr/bin/env python

"""Read CTAG2F gripper feedback through the RealMan socket JSON Modbus bridge."""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any


READ_REGISTERS = [
    ("enable", 0x0100),
    ("target_position_high", 0x0102),
    ("target_position_low", 0x0103),
    ("target_speed_percent", 0x0104),
    ("target_torque_percent", 0x0105),
    ("torque_reached", 0x0601),
    ("position_reached", 0x0602),
    ("speed_reached", 0x0603),
    ("ready", 0x0604),
    ("realtime_position_high", 0x0609),
    ("realtime_position_low", 0x060A),
    ("realtime_user_position_high", 0x060D),
    ("realtime_user_position_low", 0x060E),
    ("realtime_speed", 0x060B),
    ("realtime_current", 0x060C),
    ("alarm", 0x0612),
    ("parameter_modified", 0x0614),
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


def request(host: str, port: int, payload: dict[str, Any], timeout_s: float) -> dict[str, Any] | str:
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall((json.dumps(payload, separators=(",", ":")) + "\r\n").encode())
            buffer = ""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    return "<empty>"
                buffer += chunk.decode("utf-8", errors="replace")
                for raw_line in buffer.splitlines():
                    line = raw_line.strip()
                    if line:
                        return json.loads(line)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def decode_register_data(response: dict[str, Any] | str) -> int | list[int] | None:
    if not isinstance(response, dict):
        return None
    data = response.get("data")
    if isinstance(data, list):
        return [int(value) for value in data]
    if isinstance(data, int | float):
        return int(data)
    return None


def combine_u32(high: int | list[int] | None, low: int | list[int] | None) -> int | None:
    if isinstance(high, list) or isinstance(low, list):
        return None
    if isinstance(high, int) and isinstance(low, int):
        return (high << 16) + (low & 0xFFFF)
    return None


def read_register(
    client: JsonSocket,
    args: argparse.Namespace,
    name: str,
    address: int,
) -> tuple[str, dict[str, Any] | str, int | list[int] | None]:
    payload: dict[str, Any] = {
        "command": "read_holding_registers",
        "port": args.modbus_port,
        "address": address,
        "device": args.device,
    }
    response = client.request(payload)
    return name, response, decode_register_data(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=["192.168.1.19", "192.168.1.20"])
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--modbus-port", type=int, default=1)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--modbus-timeout", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Do not send set_modbus_mode before reading registers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for host in args.hosts:
        print(f"\n== {host} CTAG2F Modbus feedback ==")
        try:
            client = JsonSocket(host, args.arm_port, args.timeout_s)
        except Exception as exc:
            print(f"connect ERROR: {type(exc).__name__}: {exc}")
            continue
        try:
            if not args.skip_init:
                init_response = client.request(
                    {
                        "command": "set_modbus_mode",
                        "port": args.modbus_port,
                        "baudrate": args.baudrate,
                        "timeout": args.modbus_timeout,
                    }
                )
                print("set_modbus_mode", init_response)

            for sample_idx in range(args.samples):
                print(f"-- sample {sample_idx} --")
                decoded: dict[str, int | list[int] | None] = {}
                for name, address in READ_REGISTERS:
                    reg_name, response, data = read_register(client, args, name, address)
                    decoded[reg_name] = data
                    print(f"{reg_name} 0x{address:04X}/{address}: raw={data} response={response}")
                decoded["target_position"] = combine_u32(
                    decoded["target_position_high"], decoded["target_position_low"]
                )
                decoded["realtime_position"] = combine_u32(
                    decoded["realtime_position_high"], decoded["realtime_position_low"]
                )
                decoded["realtime_user_position"] = combine_u32(
                    decoded["realtime_user_position_high"], decoded["realtime_user_position_low"]
                )
                print("decoded", decoded)
                if sample_idx < args.samples - 1:
                    time.sleep(args.interval_s)
        except Exception as exc:
            print(f"probe ERROR: {type(exc).__name__}: {exc}")
        finally:
            client.close()


if __name__ == "__main__":
    main()

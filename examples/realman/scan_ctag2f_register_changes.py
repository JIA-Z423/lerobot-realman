#!/usr/bin/env python

"""Scan CTAG2F holding registers and report values that change during manual motion."""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RegisterStats:
    first: int
    last: int
    minimum: int
    maximum: int
    changes: int = 0

    def update(self, value: int) -> bool:
        changed = value != self.last
        if changed:
            self.changes += 1
        self.last = value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        return changed


class JsonSocket:
    def __init__(self, host: str, port: int, timeout_s: float):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.settimeout(timeout_s)
        self.buffer = ""

    def close(self) -> None:
        self.sock.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        expected_command = payload["command"]
        self.sock.sendall((json.dumps(payload, separators=(",", ":")) + "\r\n").encode())
        while True:
            while "\n" in self.buffer:
                raw_line, self.buffer = self.buffer.split("\n", 1)
                line = raw_line.strip()
                if not line:
                    continue
                response = json.loads(line)
                if response.get("command") == expected_command:
                    return response

            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("RealMan controller closed the connection.")
            self.buffer += chunk.decode("utf-8", errors="replace")


def u16_to_i16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


def decode_register_bytes(data: Any, count: int) -> list[int]:
    if not isinstance(data, list) or len(data) != count * 2:
        raise RuntimeError(f"Expected {count * 2} data bytes, received {data!r}.")
    return [((int(data[i]) & 0xFF) << 8) | (int(data[i + 1]) & 0xFF) for i in range(0, len(data), 2)]


def read_registers(
    client: JsonSocket,
    modbus_port: int,
    device: int,
    start: int,
    count: int,
) -> list[int]:
    if count >= 3:
        response = client.request(
            {
                "command": "read_multiple_holding_registers",
                "port": modbus_port,
                "address": start,
                "num": count,
                "device": device,
            }
        )
        if response.get("read_state") is False:
            raise RuntimeError(f"Failed to read 0x{start:04X}-0x{start + count - 1:04X}: {response}")
        return decode_register_bytes(response.get("data"), count)

    values = []
    for address in range(start, start + count):
        response = client.request(
            {
                "command": "read_holding_registers",
                "port": modbus_port,
                "address": address,
                "device": device,
            }
        )
        if response.get("read_state") is False or not isinstance(response.get("data"), int | float):
            raise RuntimeError(f"Failed to read 0x{address:04X}: {response}")
        values.append(int(response["data"]) & 0xFFFF)
    return values


def discover_read_plan(
    client: JsonSocket,
    args: argparse.Namespace,
    start: int,
    count: int,
) -> tuple[list[tuple[int, int]], dict[int, int]]:
    try:
        block_values = read_registers(client, args.modbus_port, args.device, start, count)
    except RuntimeError:
        if count == 1:
            print(f"Skipping unreadable register 0x{start:04X}")
            return [], {}

        left_count = count // 2
        right_count = count - left_count
        left_plan, left_values = discover_read_plan(client, args, start, left_count)
        right_plan, right_values = discover_read_plan(client, args, start + left_count, right_count)
        return left_plan + right_plan, {**left_values, **right_values}

    return [(start, count)], {start + offset: value for offset, value in enumerate(block_values)}


def discover_range(
    client: JsonSocket, args: argparse.Namespace
) -> tuple[list[tuple[int, int]], dict[int, int]]:
    plan: list[tuple[int, int]] = []
    values: dict[int, int] = {}
    address = args.start_address
    while address <= args.end_address:
        count = min(args.batch_size, args.end_address - address + 1)
        block_plan, block_values = discover_read_plan(client, args, address, count)
        plan.extend(block_plan)
        values.update(block_values)
        address += count
    return plan, values


def read_plan(client: JsonSocket, args: argparse.Namespace, plan: list[tuple[int, int]]) -> dict[int, int]:
    values: dict[int, int] = {}
    for start, count in plan:
        block_values = read_registers(client, args.modbus_port, args.device, start, count)
        values.update({start + offset: value for offset, value in enumerate(block_values)})
    return values


def parse_int(value: str) -> int:
    return int(value, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--modbus-port", type=int, default=1)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--modbus-timeout", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--start-address", type=parse_int, default=0x0600)
    parser.add_argument("--end-address", type=parse_int, default=0x0614)
    parser.add_argument("--batch-size", type=int, choices=range(3, 13), default=12)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--interval-s", type=float, default=0.05)
    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument(
        "--leave-modbus-open",
        action="store_true",
        help="Do not restore the RS485 port to its previous mode when scanning finishes.",
    )
    parser.add_argument(
        "--restore-mode",
        action="store_true",
        help="Only close Modbus mode to restore hardware control, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_address < args.start_address:
        raise ValueError("--end-address must not be smaller than --start-address.")

    client = JsonSocket(args.host, args.arm_port, args.timeout_s)
    configured_modbus = False
    try:
        if args.restore_mode:
            response = client.request({"command": "close_modbus_mode", "port": args.modbus_port})
            print("close_modbus_mode", response)
            return

        if not args.skip_init:
            response = client.request(
                {
                    "command": "set_modbus_mode",
                    "port": args.modbus_port,
                    "baudrate": args.baudrate,
                    "timeout": args.modbus_timeout,
                }
            )
            print("set_modbus_mode", response)
            if response.get("set_state") is False:
                raise RuntimeError(f"Failed to configure Modbus: {response}")
            configured_modbus = True

        read_blocks, initial = discover_range(client, args)
        if not initial:
            raise RuntimeError("No readable registers were found in the requested range.")
        print(
            "Readable blocks: "
            + ", ".join(f"0x{start:04X}-0x{start + count - 1:04X}" for start, count in read_blocks)
        )
        stats = {address: RegisterStats(value, value, value, value) for address, value in initial.items()}
        print(
            f"Scanning 0x{args.start_address:04X}-0x{args.end_address:04X} for "
            f"{args.duration_s:g}s. Move the gripper continuously now."
        )

        deadline = time.monotonic() + args.duration_s
        sample = 0
        while time.monotonic() < deadline:
            sample += 1
            try:
                values = read_plan(client, args, read_blocks)
            except RuntimeError as exc:
                print(f"sample {sample:04d}: transient read failure, skipping ({exc})")
                continue
            changed = []
            for address, value in values.items():
                if stats[address].update(value):
                    changed.append(f"0x{address:04X}={value} (signed={u16_to_i16(value)})")
            if changed:
                print(f"sample {sample:04d}: " + ", ".join(changed))
            if args.interval_s > 0:
                time.sleep(args.interval_s)

        ranked = sorted(
            stats.items(), key=lambda item: (item[1].changes, item[1].maximum - item[1].minimum), reverse=True
        )
        print("\nChanged-register summary:")
        for address, stat in ranked:
            if stat.changes:
                print(
                    f"0x{address:04X}: changes={stat.changes}, "
                    f"unsigned_range=[{stat.minimum}, {stat.maximum}], "
                    f"signed_range=[{u16_to_i16(stat.minimum)}, {u16_to_i16(stat.maximum)}], "
                    f"first={stat.first}, last={stat.last}"
                )
    finally:
        if configured_modbus and not args.leave_modbus_open:
            try:
                print(
                    "close_modbus_mode",
                    client.request({"command": "close_modbus_mode", "port": args.modbus_port}),
                )
            except Exception as exc:
                print(f"WARNING: failed to restore RS485 mode: {exc}")
        client.close()


if __name__ == "__main__":
    main()

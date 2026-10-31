#!/usr/bin/env python

"""Read CTAG2F Modbus registers through the RealMan vendor SDK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_VENDOR_ROOT = Path("/home/nuaacmee/rm_aloha_v1")

REGISTERS = [
    ("enable", 0x0100),
    ("target_high", 0x0102),
    ("target_low", 0x0103),
    ("target_speed_percent", 0x0104),
    ("target_torque_percent", 0x0105),
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


def combine(high: int | None, low: int | None) -> int | None:
    if high is None or low is None:
        return None
    return (high << 16) + (low & 0xFFFF)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--modbus-port", type=int, default=1)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--modbus-timeout", type=int, default=2)
    parser.add_argument("--skip-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.vendor_root))
    from RM_robotic.robotic_arm_package.robotic_arm import RM65, Arm

    arm = Arm(RM65, args.host)
    try:
        print("Arm_Socket_State", arm.Arm_Socket_State())
        if not args.skip_init:
            print(
                "Set_Modbus_Mode",
                arm.Set_Modbus_Mode(args.modbus_port, args.baudrate, args.modbus_timeout, False),
            )
        values: dict[str, int | None] = {}
        for name, address in REGISTERS:
            result, value = arm.Get_Read_Holding_Registers(args.modbus_port, address, args.device)
            values[name] = value if result == 0 else None
            print(f"{name} 0x{address:04X}/{address}: result={result} value={value}")
        values["target"] = combine(values["target_high"], values["target_low"])
        values["position"] = combine(values["pos_high"], values["pos_low"])
        values["user_position"] = combine(values["user_pos_high"], values["user_pos_low"])
        print("decoded", values)
        for name, address, num in [
            ("target_pair", 0x0102, 2),
            ("position_pair", 0x0609, 2),
            ("user_position_pair", 0x060D, 2),
        ]:
            result, data = arm.Read_Multiple_Holding_Registers(
                args.modbus_port, address, num, args.device
            )
            print(f"{name} 0x{address:04X}/{address} num={num}: result={result} data={data}")
    finally:
        try:
            arm.Arm_Socket_Close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

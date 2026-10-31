#!/usr/bin/env python

"""Read-only RealMan vendor SDK status probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_VENDOR_ROOT = Path("/home/nuaacmee/rm_aloha_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=["192.168.1.19", "192.168.1.20"])
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.vendor_root))

    from RM_robotic.robotic_arm_package.robotic_arm import RM65, Arm

    read_methods = [
        "Arm_Socket_State",
        "Get_Tool_Voltage",
        "Get_Tool_IO_State",
        "Get_Voltage",
        "Get_Arm_Power_State",
        "Get_Arm_Software_Version",
    ]

    for host in args.hosts:
        print(f"\n== {host} vendor sdk read-only ==")
        arm = Arm(RM65, host)
        try:
            for name in read_methods:
                method = getattr(arm, name)
                try:
                    print(name, method())
                except Exception as exc:
                    print(name, f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            try:
                arm.Arm_Socket_Close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

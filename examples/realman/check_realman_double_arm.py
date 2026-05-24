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

"""Hardware smoke check for the RealMan double-arm passive recording setup.

This script only reads state from the robot arms, master arms, and cameras. It
does not send joint targets and it does not modify the YAML configuration.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if REPO_SRC.exists():
    sys.path.insert(0, str(REPO_SRC))


DEFAULT_JOINT_SCALE = 1000.0
DEFAULT_GRIPPER_SCALE = 1000.0


@dataclass(frozen=True)
class ArmEndpoint:
    name: str
    host: str
    port: int


@dataclass(frozen=True)
class MasterEndpoint:
    name: str
    port: str


@dataclass(frozen=True)
class CameraEndpoint:
    name: str
    serial: str


def _request_json(
    endpoint: ArmEndpoint,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    expected_key: str | None = None,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        message = json.dumps(payload, separators=(",", ":")) + "\r\n"
        sock.sendall(message.encode("utf-8"))
        buffer = ""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError(f"{endpoint.name}: empty response for {payload}")
            buffer += chunk.decode("utf-8")
            for raw_line in buffer.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                response = json.loads(line)
                if expected_key is None or expected_key in response:
                    return response, time.perf_counter() - start


def check_arm(endpoint: ArmEndpoint, *, timeout_s: float, check_gripper: bool) -> bool:
    ok = True
    print(f"\n[{endpoint.name} follower] {endpoint.host}:{endpoint.port}")
    try:
        response, elapsed_s = _request_json(
            endpoint,
            {"command": "get_joint_degree"},
            timeout_s=timeout_s,
            expected_key="joint",
        )
        joints = response.get("joint")
        if not isinstance(joints, list):
            raise RuntimeError(f"unexpected joint response: {response}")
        scaled = [float(value) / DEFAULT_JOINT_SCALE for value in joints]
        print(f"  get_joint_degree: ok, {len(joints)} values, {elapsed_s * 1000:.1f} ms")
        print(f"  joints/1000: {scaled}")
    except Exception as exc:
        ok = False
        print(f"  get_joint_degree: FAILED: {exc}")

    if check_gripper:
        try:
            response, elapsed_s = _request_json(
                endpoint,
                {"command": "get_gripper_state"},
                timeout_s=timeout_s,
                expected_key="actpos",
            )
            raw_actpos = response.get("actpos")
            if raw_actpos is None:
                raise RuntimeError(f"unexpected gripper response: {response}")
            scaled_actpos = float(raw_actpos) / DEFAULT_GRIPPER_SCALE
            print(
                f"  get_gripper_state: ok, {elapsed_s * 1000:.1f} ms, "
                f"raw actpos={raw_actpos}, scaled={scaled_actpos}"
            )
        except Exception as exc:
            ok = False
            print(f"  get_gripper_state: FAILED: {exc}")
    return ok


def _format_action(action: dict[str, float]) -> list[float]:
    return [float(action[f"joint_{idx}.pos"]) for idx in range(1, 7)] + [float(action["gripper.pos"])]


def check_master(
    endpoint: MasterEndpoint,
    *,
    baud_rate: int,
    timeout_s: float,
    frame_timeout_s: float,
    frame_header: str,
) -> bool:
    print(f"\n[{endpoint.name} master] {endpoint.port}")
    teleop = None
    try:
        from lerobot.teleoperators.rm_aloha_master import RMAlohaMaster, RMAlohaMasterConfig

        teleop = RMAlohaMaster(
            RMAlohaMasterConfig(
                id=f"realman_check_{endpoint.name}",
                port=endpoint.port,
                baud_rate=baud_rate,
                timeout_s=timeout_s,
                frame_timeout_s=frame_timeout_s,
                frame_header=frame_header,
            )
        )
        teleop.connect()
        start = time.perf_counter()
        action = teleop.get_action()
        elapsed_s = time.perf_counter() - start
        values = _format_action(action)
        print(f"  frame: ok, {len(values)} values, {elapsed_s * 1000:.1f} ms")
        print(f"  joints + gripper: {values}")
        return True
    except Exception as exc:
        print(f"  frame: FAILED: {exc}")
        return False
    finally:
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()


def check_camera(
    endpoint: CameraEndpoint,
    *,
    width: int,
    height: int,
    fps: int,
    max_age_ms: int,
) -> bool:
    print(f"\n[{endpoint.name} camera] serial={endpoint.serial}")
    camera = None
    try:
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

        camera = RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=endpoint.serial,
                width=width,
                height=height,
                fps=fps,
            )
        )
        camera.connect()
        start = time.perf_counter()
        frame = camera.read_latest(max_age_ms=max_age_ms)
        elapsed_s = time.perf_counter() - start
        nonzero = bool(frame.any())
        expected_shape = (height, width, 3)
        shape_ok = tuple(frame.shape) == expected_shape
        print(f"  frame: ok, shape={tuple(frame.shape)}, nonzero={nonzero}, {elapsed_s * 1000:.1f} ms")
        if not shape_ok:
            print(f"  expected shape: {expected_shape}")
        return nonzero and shape_ok
    except Exception as exc:
        print(f"  frame: FAILED: {exc}")
        return False
    finally:
        if camera is not None and camera.is_connected:
            camera.disconnect()


def benchmark_passive_loop(args: argparse.Namespace) -> bool:
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.robots.double_realman_follower import DoubleRealManFollower, DoubleRealManFollowerConfig
    from lerobot.robots.realman_follower import RealManFollowerBaseConfig
    from lerobot.teleoperators.double_rm_aloha_master import DoubleRMAlohaMaster, DoubleRMAlohaMasterConfig
    from lerobot.teleoperators.rm_aloha_master import RMAlohaMasterBaseConfig

    robot = DoubleRealManFollower(
        DoubleRealManFollowerConfig(
            id="realman_double_arm_benchmark",
            left_arm_config=RealManFollowerBaseConfig(
                host=args.left_host,
                port=args.arm_port,
                has_gripper=args.check_gripper,
                gripper_scale=DEFAULT_GRIPPER_SCALE,
                cameras={
                    "wrist": RealSenseCameraConfig(
                        serial_number_or_name=args.left_wrist_serial,
                        width=args.camera_width,
                        height=args.camera_height,
                        fps=args.camera_fps,
                    )
                },
            ),
            right_arm_config=RealManFollowerBaseConfig(
                host=args.right_host,
                port=args.arm_port,
                has_gripper=args.check_gripper,
                gripper_scale=DEFAULT_GRIPPER_SCALE,
                cameras={
                    "wrist": RealSenseCameraConfig(
                        serial_number_or_name=args.right_wrist_serial,
                        width=args.camera_width,
                        height=args.camera_height,
                        fps=args.camera_fps,
                    )
                },
            ),
        )
    )
    teleop = DoubleRMAlohaMaster(
        DoubleRMAlohaMasterConfig(
            id="realman_double_master_benchmark",
            left_arm_config=RMAlohaMasterBaseConfig(
                port=args.left_master_port,
                baud_rate=args.master_baud_rate,
                timeout_s=args.master_timeout_s,
                frame_timeout_s=args.master_frame_timeout_s,
                frame_header=args.master_frame_header,
            ),
            right_arm_config=RMAlohaMasterBaseConfig(
                port=args.right_master_port,
                baud_rate=args.master_baud_rate,
                timeout_s=args.master_timeout_s,
                frame_timeout_s=args.master_frame_timeout_s,
                frame_header=args.master_frame_header,
            ),
        )
    )

    obs_times: list[float] = []
    action_times: list[float] = []
    total_times: list[float] = []
    try:
        robot.connect()
        teleop.connect()
        for _ in range(args.benchmark_samples):
            start = time.perf_counter()
            obs_start = time.perf_counter()
            robot.get_observation()
            obs_times.append(time.perf_counter() - obs_start)
            action_start = time.perf_counter()
            teleop.get_action()
            action_times.append(time.perf_counter() - action_start)
            total_times.append(time.perf_counter() - start)
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()

    avg_total = mean(total_times)
    print("\n[passive loop benchmark]")
    print(f"  samples: {args.benchmark_samples}")
    print(f"  observation avg: {mean(obs_times) * 1000:.1f} ms")
    print(f"  action avg: {mean(action_times) * 1000:.1f} ms")
    print(f"  total avg: {avg_total * 1000:.1f} ms ({1 / avg_total:.1f} Hz)")
    print(f"  total max: {max(total_times) * 1000:.1f} ms")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-host", default="REPLACE_WITH_LEFT_REALMAN_ARM_HOST")
    parser.add_argument("--right-host", default="REPLACE_WITH_RIGHT_REALMAN_ARM_HOST")
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--arm-timeout-s", type=float, default=2.0)
    parser.add_argument("--check-gripper", action="store_true")

    parser.add_argument("--left-master-port", default="REPLACE_WITH_LEFT_MASTER_SERIAL_PORT")
    parser.add_argument("--right-master-port", default="REPLACE_WITH_RIGHT_MASTER_SERIAL_PORT")
    parser.add_argument("--master-baud-rate", type=int, default=460800)
    parser.add_argument("--master-timeout-s", type=float, default=0.0)
    parser.add_argument("--master-frame-timeout-s", type=float, default=0.2)
    parser.add_argument("--master-frame-header", default="AA55")

    parser.add_argument("--left-wrist-serial", default="REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL")
    parser.add_argument("--right-wrist-serial", default="REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-max-age-ms", type=int, default=1000)

    parser.add_argument(
        "--skip-cameras",
        action="store_true",
        help="Skip RealSense checks when the cameras are not connected.",
    )
    parser.add_argument(
        "--skip-masters",
        action="store_true",
        help="Skip master-arm serial checks when the master arms are not connected.",
    )
    parser.add_argument(
        "--skip-followers",
        action="store_true",
        help="Skip follower-arm TCP checks when the robot controllers are not connected.",
    )
    parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=0,
        help="Run a read-only passive loop benchmark for this many samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checks: list[bool] = []
    if not args.skip_followers:
        checks.extend(
            [
                check_arm(
                    ArmEndpoint("left", args.left_host, args.arm_port),
                    timeout_s=args.arm_timeout_s,
                    check_gripper=args.check_gripper,
                ),
                check_arm(
                    ArmEndpoint("right", args.right_host, args.arm_port),
                    timeout_s=args.arm_timeout_s,
                    check_gripper=args.check_gripper,
                ),
            ]
        )

    if not args.skip_masters:
        checks.extend(
            [
                check_master(
                    MasterEndpoint("left", args.left_master_port),
                    baud_rate=args.master_baud_rate,
                    timeout_s=args.master_timeout_s,
                    frame_timeout_s=args.master_frame_timeout_s,
                    frame_header=args.master_frame_header,
                ),
                check_master(
                    MasterEndpoint("right", args.right_master_port),
                    baud_rate=args.master_baud_rate,
                    timeout_s=args.master_timeout_s,
                    frame_timeout_s=args.master_frame_timeout_s,
                    frame_header=args.master_frame_header,
                ),
            ]
        )

    if not args.skip_cameras:
        checks.extend(
            [
                check_camera(
                    CameraEndpoint("left_wrist", args.left_wrist_serial),
                    width=args.camera_width,
                    height=args.camera_height,
                    fps=args.camera_fps,
                    max_age_ms=args.camera_max_age_ms,
                ),
                check_camera(
                    CameraEndpoint("right_wrist", args.right_wrist_serial),
                    width=args.camera_width,
                    height=args.camera_height,
                    fps=args.camera_fps,
                    max_age_ms=args.camera_max_age_ms,
                ),
            ]
        )

    if args.benchmark_samples > 0:
        checks.append(benchmark_passive_loop(args))

    if checks and all(checks):
        print("\nRealMan double-arm check passed.")
        return

    failed = len([check for check in checks if not check])
    raise SystemExit(f"\nRealMan double-arm check failed: {failed} failed check(s).")


if __name__ == "__main__":
    main()

"""Offline audit: fake transport bytes, real RealMan adapters/record loop/Parquet writer.

No hardware connections, downloads or existing dataset writes. Run with the
repository's installed Python environment; --output must be a new directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def inspect_datasets(bases):
    results = []
    for base in bases:
        for info_path in sorted(base.rglob("meta/info.json")):
            root = info_path.parent.parent
            info = json.loads(info_path.read_text())
            paths = sorted(root.glob("data/**/*.parquet"))
            item = {
                "root": str(root),
                "fps": info["fps"],
                "metadata_frames": info["total_frames"],
                "version": info["codebase_version"],
                "columns": {},
            }
            for key in ("observation.state", "action"):
                feature = info["features"][key]
                column = {"feature": feature}
                if paths:
                    tables = [pq.read_table(path, columns=[key]) for path in paths]
                    # Preserve the stored Arrow value type, rather than assuming float32.
                    arrays = [
                        np.stack(table[key].combine_chunks().to_numpy(zero_copy_only=False))
                        for table in tables
                    ]
                    values = np.concatenate(arrays)
                    column.update(
                        shape=list(values.shape),
                        dtype=str(values.dtype),
                        arrow_type=str(tables[0].schema.field(key).type),
                        zero_indices=np.flatnonzero(np.all(values == 0, axis=0)).tolist(),
                        nonfinite_count=int((~np.isfinite(values)).sum()),
                        min=values.min(axis=0).tolist(),
                        max=values.max(axis=0).tolist(),
                        mean=values.astype(np.float64).mean(axis=0).tolist(),
                        std=values.astype(np.float64).std(axis=0).tolist(),
                    )
                item["columns"][key] = column
            results.append(item)
    return results


class FakeSocket:
    def __init__(self, position, missing=False, offline=False):
        self.position, self.missing, self.offline = position, missing, offline
        self.requests = []
        self.pending = b""

    def sendall(self, data):
        payload = json.loads(data)
        self.requests.append(payload)
        command = payload["command"]
        if command == "get_joint_degree":
            response = {"joint": [1000, 2000, 3000, 4000, 5000, 6000]}
        elif command == "get_gripper_state":
            response = {"command": command, "enable": 1, "status": 0 if self.offline else 1, "error": 0}
            if not self.missing:
                response["actpos"] = self.position
        else:
            raise AssertionError(f"Unexpected control/config command: {payload}")
        self.pending = (json.dumps(response) + "\r\n").encode()

    def recv(self, size):
        data, self.pending = self.pending[:size], self.pending[size:]
        return data

    def close(self):
        pass


class FakeSerial:
    def __init__(self, position):
        self.is_open = True
        self.pending = b""
        # Use the real decoder's field offsets; this verifies transport decoding,
        # not the correctness of the hardware protocol specification itself.
        from lerobot.teleoperators.rm_aloha_master.rm_aloha_master import RMAlohaMaster

        encoded = list("AA55" + "00" * 39)
        for (start, end), value in zip(
            RMAlohaMaster._field_slices, [10000, 20000, 30000, 40000, 50000, 60000, position], strict=True
        ):
            encoded[start:end] = value.to_bytes(4, "little", signed=True).hex().upper()
        self.frame = bytes.fromhex("".join(encoded))

    @property
    def in_waiting(self):
        return len(self.pending)

    def write(self, data):
        self.pending = self.frame

    def read(self, size):
        data, self.pending = self.pending[:size], self.pending[size:]
        return data

    def close(self):
        self.is_open = False


def pipeline_case(output, name, has_gripper=True, scale=1000.0):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors
    from lerobot.robots.double_realman_follower import DoubleRealManFollower, DoubleRealManFollowerConfig
    from lerobot.robots.realman_follower import RealManFollowerBaseConfig
    from lerobot.scripts.lerobot_record import _validate_passive_recording_action_features, record_loop
    from lerobot.teleoperators.double_rm_aloha_master import DoubleRMAlohaMaster, DoubleRMAlohaMasterConfig
    from lerobot.teleoperators.rm_aloha_master import RMAlohaMasterBaseConfig
    from lerobot.utils.feature_utils import combine_feature_dicts

    robot = DoubleRealManFollower(
        DoubleRealManFollowerConfig(
            calibration_dir=output / "calibration",
            left_arm_config=RealManFollowerBaseConfig(
                host="offline-left",
                has_gripper=has_gripper,
                gripper_scale=scale,
                initialize_gripper_route=False,
            ),
            right_arm_config=RealManFollowerBaseConfig(
                host="offline-right",
                has_gripper=has_gripper,
                gripper_scale=scale,
                initialize_gripper_route=False,
            ),
        )
    )
    teleop = DoubleRMAlohaMaster(
        DoubleRMAlohaMasterConfig(
            calibration_dir=output / "calibration",
            left_arm_config=RMAlohaMasterBaseConfig(
                port="offline-left", frame_header="AA55", gripper_scale=scale
            ),
            right_arm_config=RMAlohaMasterBaseConfig(
                port="offline-right", frame_header="AA55", gripper_scale=scale
            ),
        )
    )
    sockets = [FakeSocket(321), FakeSocket(654)]
    for arm, transport in zip([robot.left_arm, robot.right_arm], sockets, strict=True):
        arm.client._socket = transport
        arm.client._is_connected = True
    for arm, value in zip([teleop.left_arm, teleop.right_arm], [123, 987], strict=True):
        arm.serial = FakeSerial(value)
    action_processor, robot_processor, observation_processor = make_default_processors()
    _validate_passive_recording_action_features(robot, teleop)
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            action_processor, create_initial_features(action=robot.action_features), use_videos=False
        ),
        aggregate_pipeline_dataset_features(
            observation_processor,
            create_initial_features(observation=robot.observation_features),
            use_videos=False,
        ),
    )
    dataset = LeRobotDataset.create(
        repo_id=f"audit/{name}",
        fps=30,
        root=output / name,
        features=features,
        robot_type=robot.name,
        use_videos=False,
    )
    expected = {
        "observation.state": np.array([321, 654]) / scale if has_gripper else np.zeros(2),
        "action": np.array([123, 987]) / scale,
    }
    indices = {
        key: [dataset.features[key]["names"].index(f"{side}_gripper.pos") for side in ["left", "right"]]
        for key in expected
    }
    frames = []
    events = {"exit_early": False}
    original_add = dataset.add_frame

    def checked_add(frame):
        frames.append({key: frame[key][indices[key]].tolist() for key in expected})
        for key in expected:
            np.testing.assert_allclose(frame[key][indices[key]], expected[key], rtol=1e-6)
        original_add(frame)
        for key in expected:
            np.testing.assert_allclose(
                dataset.writer.episode_buffer[key][-1][indices[key]], expected[key], rtol=1e-6
            )
        if len(frames) == 3:
            events["exit_early"] = True

    dataset.add_frame = checked_add
    try:
        record_loop(
            robot=robot,
            events=events,
            fps=30,
            teleop_action_processor=action_processor,
            robot_action_processor=robot_processor,
            robot_observation_processor=observation_processor,
            dataset=dataset,
            teleop=teleop,
            control_time_s=5,
            single_task="offline audit",
            passive_recording=True,
        )
        assert len(frames) == 3
        dataset.save_episode()
        dataset.finalize()
        table = pq.read_table(next((output / name / "data").rglob("*.parquet")))
        persisted = {}
        for key in expected:
            values = np.array(table[key].to_pylist())[:, indices[key]]
            np.testing.assert_allclose(values, np.tile(expected[key], (3, 1)), rtol=1e-6)
            persisted[key] = values.tolist()
        requests = [[r["command"] for r in sock.requests] for sock in sockets]
        assert all(set(r) <= {"get_joint_degree", "get_gripper_state"} for r in requests)
        assert all(("get_gripper_state" in r) == has_gripper for r in requests)
        return {
            "case": name,
            "indices": indices,
            "frames": frames,
            "parquet": persisted,
            "requests": requests,
            "passed": True,
        }
    finally:
        dataset.finalize()
        robot.disconnect()
        teleop.disconnect()


def boundary_cases():
    from lerobot.common.realman.client import RealManClientConfig, RealManSocketClient

    results = []
    for name, transport in [
        ("missing_actpos", FakeSocket(321, missing=True)),
        ("offline_status_zero", FakeSocket(0, offline=True)),
    ]:
        client = RealManSocketClient(RealManClientConfig(host="offline"))
        client._socket, client._is_connected = transport, True
        try:
            value = client.get_joint_positions()["gripper"]
            assert name == "offline_status_zero" and value == 0
            results.append({"case": name, "value": value, "finding": "status=0 is accepted"})
        except RuntimeError as exc:
            assert name == "missing_actpos"
            results.append({"case": name, "error": str(exc), "finding": "missing field raises; no zero fill"})
        finally:
            client.disconnect()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, nargs="*", default=[Path("datasets")])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"datasets": inspect_datasets(args.datasets)}
    result["mock"] = [
        pipeline_case(args.output, "normalized"),
        pipeline_case(args.output, "unscaled", scale=1.0),
        pipeline_case(args.output, "gripper_disabled", has_gripper=False),
    ]
    result["boundary"] = boundary_cases()
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"datasets": len(result["datasets"]), "mock": result["mock"], "boundary": result["boundary"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

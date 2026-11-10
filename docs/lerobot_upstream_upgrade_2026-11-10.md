# LeRobot 上游升级验证（2026-11-10）

## 当前版本和隔离方式

- 当前开发分支：`dev`。本次首先将原有改动提交为 `04fb806`（完善 RealMan 只读检查摘要和操作文档）。
- 当前源码 `pyproject.toml` 和原 Conda 环境安装元数据均为 `0.5.2`；Python 为 `3.12.13`。由于本仓库以整树导入开始，不能据此还原准确的原始上游提交。
- 核查到的上游最新稳定版：`v0.7.0`，2026-11-05 发布，提交 `410c095eacc0011d998a73383eb94abed2715af0`。GitHub Release、Git tag 和 PyPI 均一致。
- 升级候选分支：`upgrade/lerobot-v0.7.0`，工作区 `/tmp/lerobot-upgrade-v0.7.0`。当前仓库继续留在 `dev`。
- 候选以当前开发提交为第一父提交、稳定版标签为第二父提交接入上游历史，再应用稳定版文件并移植本地适配；不使用冲突自动覆盖本地代码。初次 fetch 仅获取稳定版快照，上游历史为浅历史，后续需要时可以补全。
- 原 Conda 环境没有 `uv`。本次使用已有解释器创建候选专用 `.venv`，通过 `--system-site-packages` 只读复用现有 PyTorch/视频等大型依赖，在候选环境内安装所需的新版本。没有在原 Conda 环境原位升级。

官方来源：[GitHub Release](https://github.com/huggingface/lerobot/releases/tag/v0.7.0)、[PyPI](https://pypi.org/project/lerobot/)、[发布介绍](https://huggingface.co/blog/lerobot/lerobot-release-v070)。

## 新增模型

按配置类的源码注册装饰器核对，原树有 15 个 policy 注册名，新版有 29 个 policy 注册名和 4 个独立 reward 注册名。这是源码能力清单，不表示所有模型的依赖、权重或训练均已验证。

| 变化 | 配置类型 |
| --- | --- |
| 新增的 VLA / 世界动作模型 | `dm05`、`eo1`、`evo1`、`fastwam`、`fineart_vla`、`flux3`、`g05`、`lawam`、`lingbot_va`、`lingbot_vla_v2`、`longwam`、`molmoact2`、`vla_jepa`、`wall_oss_05`、`cosmos3_nano`、`cosmos3_edge` |
| RL 结构调整 | 原 `sac` 进入算法配置，policy 改为 `gaussian_actor` |
| 奖励模型独立 | `reward_classifier`、`sarm` 移至 rewards，新增 `robometer`、`topreward` |

`cosmos3` 是目录和 extra 名，实际 policy 类型为 `cosmos3_nano` / `cosmos3_edge`。新增模型依赖共享 VLA、处理器、配置与推理接口，不适合只复制模型目录到旧分支。

## 已移植的 RealMan 适配

- 保留 TCP JSON 客户端、RM ALOHA SDK 包装、单/双从臂与单/双主手实现。
- 将四个硬件实现的 `lerobot.types` 导入改为新版 `lerobot.lerobot_types`。
- 恢复 robot / teleop 工厂，以及采集、校准、遥操作、回放、关节限位和新版 rollout 的配置注册入口。
- 在新版采集实现中恢复 `passive_recording`：创建设备前拒绝初始化/校准，创建数据集前校验主从动作字段，录制和重置阶段都不调用 `robot.send_action`。
- 保留 RM+ 非法反馈报错、合法零值和拒绝发送保护；没有改动传输协议或硬件参数。
- 保持 action / observation.state 各 14 维，左侧在前、右侧在后；夹爪索引为 `6`、`13`。
- 保留本地文档、公开硬件模板、检查脚本和离线测试；上游 README 更新保存至候选的 `UPSTREAM_LEROBOT_README.md`。

升级候选中的模板已迁移为：

```yaml
dataset:
  rgb_encoder:
    vcodec: h264
  no_stamp: true
passive_recording: true
```

原有 `dataset.vcodec` 在新版应改为 `dataset.rgb_encoder.vcodec`。`no_stamp: true` 用于保留手动管理的数据集标识；新版默认会为新录制的 repo_id 添加时间后缀。真实 `.local.yaml` 没有自动复制或修改，需在候选工作区单独准备迁移后的副本。

## 验证证据

原 `dev` 基线：

- 5 个 RealMan 离线测试文件和动作 schema 测试：`189 passed, 1 skipped`；跳过项为主手没有发送动作接口的预期跳过。
- 模拟采集审计：5 个场景各写入并回读 3 帧，全部通过。覆盖归一化、原始量纲、Modbus、RM+ 和禁用夹爪。
- RM+ 从臂反馈 `[0.321, 0.654]` 与主手 action `[0.123, 0.987]` 分别正确保存，夹爪索引保持 `6/13`；缺失/离线反馈报错，没有填零。
- 审计路径 `/tmp/realman-upstream-baseline-20261110/results.json`。审计禁止真实 socket/serial 连接，未扫描现有数据集。

升级候选：

- 同样的 RealMan 回归、迁移的 schema 测试、新增完整被动录制/重置测试，以及新版 `tests/test_control_robot.py`：**`197 passed, 1 skipped`，CPU，3.95 秒**。
- 新增的完整流程测试在任何 `send_action` 调用时立即失败，实际执行两个 episode 和中间重置，并回读全部 Parquet 帧验证动作、状态、episode 和帧序号。
- 5 个模拟采集场景全部通过；候选 `results.json` 与基线 JSON 解析后逐项比较，**全部字段完全相等**。候选证据 `/tmp/realman-upstream-candidate-20261110/results.json`。
- 公开硬件 YAML 使用新版 draccus 解码成功，确认被动模式、双臂/双主手类型、H.264 编码及固定 repo_id 设置；没有创建设备。
- 运行时成功加载 29 个 policy 和 4 个 reward 注册类型；没有加载模型权重。
- RealMan 移植涉及的源文件及测试通过 Ruff 检查、格式检查和 `git diff --check`。
- 候选专用环境安装元数据为 `lerobot=0.7.0`，`huggingface-hub=1.33.0`、`draccus=0.11.6`、`transformers=5.22.2`、`diffusers=0.39.0`、`accelerate=1.15.0`、`filelock=4.2.10`；只读复用 `torch=2.10.0`。
- 安装完成后重新读取原 Conda 环境：LeRobot、Hub、draccus、Transformers、PyTorch 仍为原版本，源码仍指向原 `dev` 工作区。
- **环境限制**：候选的 `pip check` 返回非零，唯一报告项为从原环境继承的开发工具 `virtualenv=21.2.4` 要求 `filelock<4`，而新版 LeRobot 要求 `filelock>=4.2.1`。运行回归已通过，但这种共享依赖环境不能当作完整依赖验收；正式使用应在干净环境中按锁文件安装所需 extras，并处理开发工具约束。未改动原环境来消除此冲突。

## 复现候选离线验证

```bash
cd /tmp/lerobot-upgrade-v0.7.0
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/common/test_realman_client.py \
  tests/common/test_realman_reconnect.py \
  tests/common/test_realman_readonly.py \
  tests/common/test_realman_gripper_watch.py \
  tests/common/test_realman_passive_recording.py \
  tests/common/test_realman_action_schema.py \
  tests/test_control_robot.py -q -p no:cacheprovider

# 输出目录必须尚不存在，--datasets 空列表避免扫描现有采集数据。
PYTHONPATH=src .venv/bin/python examples/realman/audit_offline_pipeline.py \
  --output /tmp/realman-upstream-candidate-new-run --datasets
```

## 切换使用前的边界

旧依赖不能直接运行新源码：原环境中 `huggingface-hub=1.11.0`、`draccus=0.10.0`、`transformers=5.3.0`，新版分别要求 `>=1.31,<3`、`>=0.11.6,<0.12`、`>=5.22,<5.23`。候选验证保留现有 `torch=2.10.0`，没有安装新版默认的 CUDA 13 PyTorch 构建。

训练还涉及 W&B → Trackio、FSDP2/HSDP、dtype / PolicyOutput 和部分旧断点恢复方式的变化。真实相机采集、双路视频持续 30 FPS、GPU 训练、新 policy 权重和实机闭环均需各自验证。本次离线验证不等于这些能力的实机验收。

当前 `dev` 及原环境继续用于集成开发；升级候选通过验收后再决定是否合入。代码提交保存在 Git 分支中，`/tmp` 环境与审计产物需在系统清理前另行保存。

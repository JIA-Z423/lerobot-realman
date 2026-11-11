# RealMan 双臂采集检查与夹爪反馈说明

实机检查：2026-11-02；软件复核：2026-11-03；操作文档整理：2026-11-06。本文汇总本机 RealMan 双臂接入 LeRobot 的检查结论，替代此前按日期分散保存的审计报告。重复日志、临时数据集和一次性动作脚本已清理。

具体使用步骤统一见[操作与硬件检查手册](realman_usage.md)，本文保留修复原因、实机证据和已知限制。

## 1. 夹爪恒零的原因及修复

本机知行 CTAG2F120 通过 **RM_ARM+ 生态协议**连接机械臂末端。旧采集路径读取原生 `get_gripper_state.actpos`，该接口在本机返回 `status=0, enable=0, actpos=0`，不能作为有效位置。

实机已复现：用户观察到夹爪张开时，RM+ 位置为约 299–300，原生 `actpos` 仍为 0。当前读取链路改为：

```text
get_rm_plus_state_info.state_info.pos[0] / 1000
→ left_gripper.pos / right_gripper.pos
→ observation.state[6] / observation.state[13]
→ Parquet
```

比例来自实机 `base_info.pos_low=[0]、pos_up=[1000]`。归一化位置无量纲，不是毫米或角度。当前测试范围内，张开约为 0.299–0.300，关闭回到 0；全程闭合时采到全零可以是正常现象。

读取失败、反馈缺失或系统/自由度错误会明确报错，不回退到原生接口、旧值或零。旧数据不会被自动修复，也不能用主手 `action` 或动作目标替代从臂反馈。

历史 `has_gripper=false` 配置还曾直接产生占位零。旧 episode 没有保留当时原始回包，不能把所有历史零值逐一归因于本次接口错配。

## 2. 当前设备和配置

| 项目             | 左侧                 | 右侧                 |
| ---------------- | -------------------- | -------------------- |
| 从臂             | `192.168.1.19:8080`  | `192.168.1.20:8080`  |
| 机械臂 / 控制器  | RM65-6FB / V1.7.4    | RM65-6FB / V1.7.4    |
| RM+ 夹爪身份     | ZX，单自由度，V1.154 | ZX，单自由度，V1.154 |
| 主手串口         | `/dev/ttyUSB0`       | `/dev/ttyUSB1`       |
| FTDI 稳定标识    | `A9ZM2XJU`           | `A966NBKW`           |
| 腕部 D435 序列号 | `346522073754`       | `347622070942`       |

主手串口为 460800 baud；USB 名称变化时，可按已确认的物理身份改用 `/dev/serial/by-id/` 路径。上表主手左右沿用本地配置，尚未通过主手动作独立标定。

使用本地 `configs/hardware/realman_double_arm_hardware_config.local.yaml`，公开参考为[配置模板](../configs/hardware/realman_double_arm_hardware_config.yaml)。两从臂保持：

```yaml
sdk_backend: socket_json
has_gripper: true
gripper_feedback_source: rm_plus
gripper_scale: 1000.0
initialize_gripper_route: false
initialize_gripper_modbus: false
```

顶层保持 `passive_recording: true`。正式采集由外部 RealMan 系统负责主从控制，LeRobot 读取并保存数据。RM+ 通用动作接口仍有拒绝发送保护；采集启动前也会拒绝夹爪初始化或校准配置。

当前没有顶部相机。两个腕部 D435 曾完全无法枚举，重新上电后均恢复 USB 3.2 识别，并通过双路 640×480、30 fps RGB 同时取帧检查。没有修改序列号；此前不可见的具体供电、线束或 Hub 原因未确定。

## 3. 已完成的真实验证

| 检查                       | 结果                                           |
| -------------------------- | ---------------------------------------------- |
| 左、右实际张开             | 用户分别目视确认；RM+ 反馈约 299–300           |
| 逐侧关闭                   | 各经用户授权，反馈逐步下降至 0，直接寄存器一致 |
| 检查结束状态               | 两侧位置、直接位置寄存器均为 0                 |
| 真实采集                   | 360 帧，state/action 均为 `[360, 14]`          |
| 小目标采集中的夹爪范围     | 两侧均为 0–0.004                               |
| 左 / 右非零帧数            | 81 / 70                                        |
| 原始回包 → state → Parquet | 按 float32 转换后逐帧、逐元素一致              |
| 采集时长 / 平均频率        | 约 12.075 秒 / 29.81 Hz，不含相机              |
| 机械臂动作指令             | 全程 0 条                                      |

360 帧录制和后续更大幅度的人工观察是两阶段测试；299–300 的动作仅用于后续观察，没有追加进这份录制。诊断录制的夹爪由独立连接控制，主手保持静止，故其 `action` 不与夹爪测试动作配对，不能作为训练示教使用。临时数据已清理，保留以上验证结果。

录制中出现过单周期约 19 Hz 的超预算提示，目标 30 fps 和平均频率不代表每个周期达标。两腕视频及编码加入后的整机性能、硬件同步、全量程精度和毫米标定尚未验收。

## 4. 采集前的只读复查

日常可在项目根目录使用既有 `lerobot` 环境，手动运行简洁检查：

```bash
conda activate lerobot
PYTHONPATH=src python examples/realman/check_realman_readonly.py --summary
```

[分项检查器](../examples/realman/check_realman_readonly.py)使用查询白名单，简洁模式显示两侧从臂、夹爪、主手和相机共 8 项结果；异常时返回非零退出码。日常无需读取 JSON，需留存原始回包时再添加 `--output`。仅使用详细报告模式、不加 `--summary` 时，退出成功仍只代表报告生成。

[实时监视器](../examples/realman/watch_realman_gripper.py)用于进一步连续观察 RM+ 模式、身份、状态及位置，不初始化或发送动作。静止时 `changed=false` 并不表示故障。

分项检查、夹爪监视和异常排查的完整命令见[手动硬件检查](realman_usage.md#4-手动硬件检查)。正式采集按[数据采集步骤](realman_usage.md#5-数据采集)执行，使用新的输出目录，避免混入历史数据。旧的原生夹爪检查器不能用于判断本机 RM+ 夹爪是否健康。

## 5. 已知限制与历史数据

- **上电恢复：** 曾出现 RM+ 模式查询为 115200，但身份和状态都读取失败。用户重新关开协议后恢复，同时工具 RS485 模式由 1 变为 0；具体启动恢复机制未证实。程序不会自动重置协议。
- **备用 Modbus 路径：** 本机未成功验收普通寄存器反馈，当前配置不启用该路径。代码按控制器要求批量读取三个寄存器，校验六字节，只用前两寄存器解析位置；设备也须支持读取紧邻的第三个寄存器。
- **限速读写：** 写末端速度限制寄存器 1180 为 50 时收到成功 ACK，但读回仍为 5000，无法确定是否为缓存或固件行为。测试结束后已回写原值 5000，查询也为 5000；实际限速语义未验收。小步目标控制不等于已验证硬件限速。
- **状态字段：** 部分位置变化期间 `dof_state` 仍为 2、速度仍为 0，不能单独据此判断精确到位。测试组合检查位置连续稳定、目标容差、寄存器及错误状态；它们仍来自同一 RM+ 反馈体系。
- **数据含义：** `action` 是主手状态，`observation.state` 是从臂反馈，均为左右各六关节加夹爪，共 14 维。夹爪索引为 6、13（从 0 开始），两者不能互相填补。
- **历史数据质量：** 已有旧数据的从臂夹爪曾恒零，主手还存在原始计数与归一化标尺差异及异常值，不能直接混训。正式历史数据保留原样。
- **时间与保存：** 数据集时间戳按帧序号和 fps 生成，相机取最新帧时可能重复图像，不等于传感器硬件同步。本地配置使用异步写图、episode 结束后 H.264 编码，数值也应等 episode 完整保存后再使用。

官方参考：[第三代末端工具 JSON](https://develop.realman-robotics.com/robot/json/endeffector/)、[RM+ 实时状态](https://develop.realman-robotics.com/robot/apic/struct/plusState/)、[RM_ARM+ 协议](https://gitee.com/RealManRobot/rm_arm_plus_protocol/blob/main/readme.md)。原始设备 PDF 手册保留在仓库根目录。

离线验证保留[完整采集管道检查脚本](../examples/realman/audit_offline_pipeline.py)及 `tests/common/test_realman_*.py` 中的反馈、只读、重连和被动录制回归测试。一次性动作探针及其快照不作为维护入口保留。

本仓库是定制的 LeRobot 源码快照。Python 包版本 0.5.2 与 Dataset v3.0 是不同版本标识，不能据此推定精确上游 tag；升级依赖不是本次修复夹爪恒零的前提。

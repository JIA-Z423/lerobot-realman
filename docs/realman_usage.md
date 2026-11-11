# RealMan 双臂操作与硬件检查手册

适用分支：`upgrade/lerobot-v0.7.0`；源码包版本：`0.7.0`；数据集格式：`v3.0`。包版本与数据集格式是不同的版本标识。

本文保存本项目的具体使用步骤：准备环境与本地配置、手动读取硬件、排查异常、采集与续录，以及检查保存的数据。所有命令默认在仓库根目录执行；本手册位于 `docs/` 下，但无需进入 `docs/` 运行命令。

日常流程：准备环境和配置 → 手动运行 `--summary` → 确认所需设备读取通过 → 被动采集 → 检查数据。检查脚本由使用者按需运行，只读取反馈和图像。

- [1. 环境安装](#1-环境安装)
- [2. 硬件参数](#2-硬件参数)
- [3. 配置文件](#3-配置文件)
- [4. 手动硬件检查](#4-手动硬件检查)
- [5. 数据采集](#5-数据采集)
- [6. 数据格式](#6-数据格式)
- [7. 数据可视化](#7-数据可视化)
- [8. 使用范围与相关资料](#8-使用范围与相关资料)
- [9. 策略训练](#9-策略训练)

## 1. 环境安装

新机器先准备 Git 和 [Conda（Miniforge）](https://conda-forge.org/download/)，获取本分支源码；已有源码时跳过此步：

```bash
git clone --branch upgrade/lerobot-v0.7.0 https://github.com/JIA-Z423/lerobot-realman.git lerobot-realman-v070
cd lerobot-realman-v070
```

请安装本仓库源码，公开 PyPI 包不包含这里的 RealMan 定制适配。以下安装和运行命令都在项目根目录执行。

在本分支根目录使用 Conda 建立独立的 Python 3.12 环境 `lerobot070`，与 `dev` 的 `lerobot` 环境分别维护。`core_scripts` 包含采集所需的数据集、硬件及可视化依赖，`intelrealsense` 提供腕部相机 SDK：

```bash
conda create -n lerobot070 python=3.12 pip -y
conda activate lerobot070
conda install -c conda-forge ffmpeg -y
python -m pip install -e ".[core_scripts,intelrealsense]"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import serial, pyrealsense2, draccus; print('ok')"
python -c "import importlib.metadata as m, lerobot; print('LeRobot', m.version('lerobot')); print(lerobot.__file__)"
```

确认打印的版本为 `0.7.0`，源码路径位于当前工作区。已有 `lerobot070` 环境时跳过创建并直接激活；后文的 `python`、`rerun` 等命令均在该 Conda 环境中运行，新终端先执行 `conda activate lerobot070`。安装后，`python -m lerobot.scripts.lerobot_record` 等模块命令也可使用 `lerobot-record` 等同名 CLI 入口。

视频处理还需要在 `PATH` 中可用的 `ffmpeg`、`ffprobe` 及解码所需的 FFmpeg 库。PyTorch/CUDA 安装选择见 [LeRobot 安装说明](source/installation.mdx)。工作区移动或重新创建后，应在新路径重新准备环境。

从 `dev` 的 `0.5.2` 迁移时，在本分支的独立 Conda 环境中安装新版依赖，并按第 3 节迁移配置。

## 2. 硬件参数

公开配置使用以下占位符。运行前请在本地 `.local.yaml` 配置中替换为真实设备参数：

| 设备     | 参数                                       |
| -------- | ------------------------------------------ |
| 左从臂   | `REPLACE_WITH_LEFT_REALMAN_ARM_HOST:8080`  |
| 右从臂   | `REPLACE_WITH_RIGHT_REALMAN_ARM_HOST:8080` |
| 左主手   | `REPLACE_WITH_LEFT_MASTER_SERIAL_PORT`     |
| 右主手   | `REPLACE_WITH_RIGHT_MASTER_SERIAL_PORT`    |
| 左腕相机 | `REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL`    |
| 右腕相机 | `REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL`   |

如果重启后串口名发生变化，建议把 `/dev/ttyUSB*` 改成稳定的设备路径：

```bash
ls -l /dev/serial/by-id/
```

## 3. 配置文件

公开[配置模板](../configs/hardware/realman_double_arm_hardware_config.yaml)使用占位符。真实设备参数保存在 `configs/hardware/realman_double_arm_hardware_config.local.yaml`，该文件已被 Git 忽略。

首次使用且尚未建立本地配置时，复制模板；已有本地配置时跳过此步：

```bash
cp -n configs/hardware/realman_double_arm_hardware_config.yaml \
  configs/hardware/realman_double_arm_hardware_config.local.yaml
```

随后编辑 `.local.yaml`：填写两侧从臂地址、主手串口、相机序列号，以及下表中的采集参数。不要将模板占位符直接用于连接设备。

以下是公开模板中的关键字段摘录，不能替代包含左右硬件参数的完整模板：

```yaml
passive_recording: true

dataset:
  repo_id: local/realman_double_arm_dataset
  single_task: RealMan double-arm passive data collection
  root: datasets/realman_double_arm_dataset
  num_episodes: 1
  fps: 30
  episode_time_s: 8
  reset_time_s: 0
  streaming_encoding: false
  rgb_encoder:
    vcodec: h264
  no_stamp: true
  push_to_hub: false

robot:
  type: double_realman_follower

teleop:
  type: double_rm_aloha_master
```

正式采集前核对：

| 字段                                                | 设置方法                                                                                   |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `dataset.repo_id`                                   | 数据集标识，如 `local/realman_double_arm_dataset`；本地保存时无需上传                      |
| `dataset.no_stamp`                                  | 模板为 `true`，新建录制保持原始 `repo_id`；配置类默认 `false` 会为新录制的标识追加时间后缀 |
| `dataset.root`                                      | 首次采集使用新目录；同一目录用于后续续录和可视化                                           |
| `dataset.single_task`                               | 填写实际任务描述，避免一直使用通用的检查说明                                               |
| `dataset.num_episodes` / `episode_time_s` / `fps`   | 分别设置采集段数、每段时长和目标采样频率                                                   |
| `dataset.streaming_encoding` / `rgb_encoder.vcodec` | 模板为 `false` / `h264`，先写图像，再在保存 episode 时编码                                 |
| `robot.left_arm_config` / `right_arm_config`        | 分别核对从臂地址、RM+ 夹爪配置和相机序列号                                                 |
| `teleop.left_arm_config` / `right_arm_config`       | 分别核对主手串口、460800 波特率及归一化比例                                                |

从旧版本迁移时核对：

| 旧配置或调用                                     | 本分支的写法与行为                                                                                   |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `dataset.vcodec: h264` / `--dataset.vcodec=h264` | 改为 YAML 中的 `dataset.rgb_encoder.vcodec` / CLI 的 `--dataset.rgb_encoder.vcodec=h264`；删除旧字段 |
| 省略 `dataset.no_stamp`                          | 如需使用本手册的固定标识，显式设为 `true`；已有带时间后缀的数据集续录时使用其实际标识                |
| 使用 `lerobot-record --policy.*` 运行策略        | 新版 `RecordConfig` 没有 `policy` 字段，采集必须有 `teleop`；策略部署入口为 `lerobot-rollout`        |

完整采集配置由 [RecordConfig](../src/lerobot/scripts/lerobot_record.py) 和共享的 [DatasetRecordConfig](../src/lerobot/configs/dataset.py) 定义。采集命令使用 `--config_path`（下划线）；分项检查器和夹爪监视器使用 `--config-path`（连字符）。硬件检查器读取 YAML 中的设备字段，检查通过不等于整个采集配置已通过 0.7.0 解析。

被动采集需要主手提供 `action`，并要求主手与从臂的动作字段完全匹配。两侧从臂的 `initialize_gripper_route`、`initialize_gripper_modbus`、`enable_calibration` 都须为 `false`；任一项为 `true` 时，采集在创建设备前报错。模板已关闭 route 初始化，其余两项的配置默认值为 `false`。

当前双臂夹爪是正式采集 schema 的一部分。公开模板和本地配置使用 `gripper_feedback_source: rm_plus`，读取 `get_rm_plus_state_info.state_info.pos[0]`，按 `gripper_scale=1000.0` 归一化；它不是毫米。默认数据为左右各 6 轴 + 夹爪，共 `14` 维 action/state。RM+ 适配当前只支持读取，发送动作会明确报错。普通原生夹爪可显式使用 `gripper_feedback_source: native`。

本机 RM+ 配置保持 `sdk_backend: socket_json`，且不设置 `gripper_position_register`。配置类会拒绝 RM+ 与 SDK 后端或普通寄存器反馈的组合；公开模板中的 SDK 切换注释不适用于当前 RM+ 夹爪配置。

配置中的 `control_hz` 仅作为兼容字段保留；被动硬件遥操作模式下，LeRobot 不控制主从臂闭环，只按 `dataset.fps` 采样主手动作、从臂状态和相机帧。

## 4. 手动硬件检查

以下命令在项目根目录、第 1 节准备并激活的 `lerobot070` Conda 环境中运行。它们查询从臂反馈、向主手串口发送状态查询或读取相机图像，不下发机械臂、夹爪目标，也不自动初始化夹爪或切换通信协议。检查前先退出占用主手串口或相机的采集程序、相机查看器。

升级前的实机检查已确认：两侧 CTAG2F120 的 RM_ARM+ 位置反馈有效；两个腕部 D435 在重新上电后均能识别，并曾同时取得 `640×480、30 fps` RGB 图像。当前没有顶部相机。**这些是历史硬件验证结果，0.7.0 的实机采集仍需复查；每次重新连接、上电后也应重新检查。**

### 4.1 手动简洁检查（日常使用）

需要检查时，在项目根目录手动执行这一条命令即可，默认读取 `.local.yaml`，检查两侧从臂、夹爪、主手和腕部相机：

```bash
PYTHONPATH=src python examples/realman/check_realman_readonly.py --summary
```

终端直接显示每项结果，不生成 JSON 文件，也不需要翻阅原始回包。输出格式如下（仅为示例，不是本次实机结果）：

```text
[通过] 左从臂：6 个关节反馈有效
[通过] 左夹爪：RM+ 位置=0.000（原始值 0，闭合时为 0 可正常）
[通过] 左主手：/dev/ttyUSB0，7 个数值读取成功
[通过] 左腕相机：640×480 RGB 取帧成功
[通过] 右从臂：6 个关节反馈有效
[通过] 右夹爪：RM+ 位置=0.000（原始值 0，闭合时为 0 可正常）
[通过] 右主手：/dev/ttyUSB1，7 个数值读取成功
[通过] 右腕相机：640×480 RGB 取帧成功
本次检查：通过 8 项，失败 0 项，跳过 0 项。
```

日常主要看是否 **8 项通过、失败 0、跳过 0**。有异常时，会显示 `[失败] 设备名：原因`，简洁模式返回非零退出码；按提示检查对应设备即可。它只确认本次静态读取可用，相机画面内容、动态跟随和持续采集性能仍需按后文确认。

只想查某一部分，可加 `--skip-masters`、`--skip-followers` 或 `--skip-cameras`，未检查项会显示 `[跳过]`。夹爪有疑问时再使用 4.2 的连续监视器；相机有疑问时再按 4.3 查看画面。

需要保留详细回包供排查时，再加输出参数；该检查器的 `--output` 会覆盖同名 JSON，需留存多次结果时使用不同文件名：

```bash
PYTHONPATH=src python examples/realman/check_realman_readonly.py \
  --summary --output /tmp/realman-readonly.json
```

下面的 JSON 字段说明供深入排查使用，日常检查无需逐项阅读。原有的详细报告模式也可继续使用：

```bash
PYTHONPATH=src python examples/realman/check_realman_readonly.py \
  --config-path configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --output /tmp/realman-readonly.json \
  --samples 3

python -m json.tool /tmp/realman-readonly.json
```

**仅使用 `--output`、未加 `--summary` 时，出现 `Read-only check saved to ...` 或退出码为 0，只代表报告已生成，不代表所有硬件通过。** 详细报告模式会将许多设备异常写入 JSON，应按需逐项查看：

| 设备           | 查看位置                                                          | 正常判据                                                                                                            |
| -------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 左、右从臂     | `followers.left/right` 的 `queries`                               | 没有连接异常；`get_joint_degree` 的 `response.joint` 各有 6 个有效数值；该查询没有 `error`                          |
| 左、右主手     | `masters.left/right`                                              | `exists=true`，`samples` 非空，每个 `state` 包含 6 个关节和 `gripper.pos` 共 7 个有效数值；设备和样本均无 `error`   |
| 左、右夹爪     | `followers.left/right.rm_plus`                                    | `enabled=true`、`errors=[]`，能够读到 `base_info` 和 `samples[].response.state_info`；还须通过 4.2 的正式解析器检查 |
| 左、右腕部相机 | `cameras.enumerated`、`cameras.configured.left_wrist/right_wrist` | 枚举列表包含配置中的两台设备；两侧均有 `frame`，尺寸为 `640×480`，没有 `error` 或 `stop_error`                      |

同时查看报告顶层及各设备的 `error`、`connection_error`、`close_error`。`skipped` 列出的设备没有参加检查，不能判为通过。分项排查可追加 `--skip-cameras`、`--skip-masters` 或 `--skip-followers`。

默认不探测普通 Modbus 寄存器；不要为了修复 RM+ 反馈而直接添加 `--probe-modbus` 或修改协议。`--probe-modbus` 只接受不带 `--summary` 的详细 JSON 模式，不能与简洁模式组合。RM+ 管理末端链路时，普通 Modbus 查询可能超时，且当前普通 Modbus 备用路径尚未完成实机验收。

### 4.2 夹爪：确认读取了正确反馈，正确理解 0

两侧从臂配置应保持以下字段；顶层保持 `passive_recording: true`：

```yaml
has_gripper: true
gripper_feedback_source: rm_plus
gripper_scale: 1000.0
initialize_gripper_route: false
initialize_gripper_modbus: false
enable_calibration: false
```

使用与正式采集相同的位置解析器，连续观察两侧夹爪 10 秒：

```bash
PYTHONPATH=src python examples/realman/watch_realman_gripper.py \
  --config-path configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --duration-s 10 \
  --fps 10
```

正常时，两侧均输出 `RM+ preflight passed`，随后持续输出 `raw=... normalized=...`。最后一行 JSON 应满足：

- `preflight_ok=true`、`interrupted=false`。
- `sides.left` 和 `sides.right` 均为 `sample_count>0`、`errors=0`。
- 原始位置在 `0..1000`，归一化位置在 `0..1`；不能有 `ERROR` 或 `PREFLIGHT ERROR`。

监视器会核对单自由度身份、位置上下限，并检查正式解析器需要的系统错误、自由度错误和位置字段。需要保留原始回包时可加 `--output /tmp/realman-gripper-watch.jsonl`；该路径必须尚不存在。预检或采样错误会返回非零退出码。

**两爪闭合时，`raw=0 normalized=0.000000` 是正常反馈；静止时 `changed=false` 也正常。** 已验证的张开位置示例为 `raw≈299–300`、`normalized≈0.299–0.300`，并不代表必须张开到这个值才算正常，归一化值也不是毫米或角度。只读检查能验证静止反馈可用，不能单凭静止数值证明全量程和动态跟随正常。

之前恒零的主要原因是读取了原生 `get_gripper_state.actpos`：本机 RM+ 夹爪实际张开时，该原生字段仍为 0。当前应读取 `get_rm_plus_state_info.state_info.pos[0] / 1000`。报告中的 `native_gripper` 仍保留原生接口对照值，不能用其 `enable=0`、`status=0` 或 `actpos=0` 单独判断 RM+ 夹爪故障；也不要改回 `native` 或关闭 `has_gripper` 来消除报错。

如果夹爪肉眼明显张开，但监视器仍显示 0，或预检失败，应保留报告继续排查。曾出现重新上电后 `mode=115200`，但身份、状态返回失败的情况；仅看到 RM_ARM+ 开关打开或模式有波特率，不足以证明设备在线。此时先在示教器中确认末端设备在线、使能及错误状态，必要时由人工重新关开 RM_ARM+ 后再运行只读复查。程序不会自动重置协议，具体上电恢复机制仍未确定。

### 4.3 腕部相机：区分未识别、序列号不匹配和取帧失败

两台腕部相机使用 RealSense 序列号识别，与主手的 `/dev/ttyUSB*` 串口号无关。只检查相机、不连接从臂 TCP 或主手串口时运行：

```bash
PYTHONPATH=src python examples/realman/check_realman_readonly.py \
  --config-path configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --output /tmp/realman-cameras-readonly.json \
  --skip-followers \
  --skip-masters

python -m json.tool /tmp/realman-cameras-readonly.json
```

按报告中的实际结果排查：

| 现象                                                           | 判断与处理                                                                                                                                            |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cameras.error` 表示无法导入 `pyrealsense2`                    | 先确认正在使用正确的 `lerobot070` Conda 环境；依赖异常不能据此判断相机未连接                                                                          |
| 枚举过程未报错但 `cameras.enumerated=[]`                       | 当前进程通过 RealSense SDK 未枚举到相机；修改 YAML 序列号无法解决。结合 `lsusb` 检查操作系统的 USB 枚举，再检查权限、驱动、供电、线束、Hub 和数据通路 |
| 枚举出设备，但出现 `Expected one matching camera ... found []` | 对照枚举得到的序列号与本地 YAML 的 `serial_number_or_name`，确认是否漏接一台、序列号不符或设备更换                                                    |
| 能匹配设备，但没有 `frame`，出现启动失败或取帧超时             | 检查相机是否被其他程序占用、USB 带宽/连接是否稳定，以及是否支持配置的 `640×480、30 fps` RGB 流；可用 `lsusb -t` 辅助检查 USB 链路速度                 |
| 两侧均有正确尺寸的 `frame`                                     | 分别取帧通过；继续目视检查画面和正式采集中的双路视频                                                                                                  |

需要检查左右视角、遮挡或黑屏时，可将两路相机图像临时保存到 `/tmp`：

```bash
camera_check_dir=$(mktemp -d /tmp/realman-camera-check.XXXXXX)
PYTHONPATH=src python -m lerobot.scripts.lerobot_find_cameras realsense \
  --record-time-s 2 \
  --output-dir "$camera_check_dir"
```

每次使用新目录，确认本轮成功保存了两台相机的图片，而非仅看到程序结束。根据终端打印的设备序列号和输出目录查看对应图片，确认视角与本地配置中的左右相符、画面可辨认；此工具使用默认流参数，不能替代上面的配置参数检查。当前没有顶部相机，本地配置中保持 `top` 注释即可。

之前两台相机曾完全无法枚举，重新上电后恢复识别，未通过修改序列号解决。具体是供电、线束还是 Hub 导致，尚未确定；相机与机械臂集成安装，也不能据此推定图像通过机械臂 TCP 或 RM_ARM+ 传输。

### 4.4 主手串口和从臂网络

主手串口重插后可能重新编号。先查看稳定设备标识，再核对本地配置中的左右主手端口及 `baud_rate: 460800`：

```bash
ls -l /dev/serial/by-id/
```

`masters.left/right.exists=false` 通常说明配置路径不存在；串口存在但读取报错时，检查权限、端口占用、波特率和实际连接。报告中的 `response_hex` 和 `state` 可用于检查帧是否解码成功。两侧各读出 7 个有效数值说明通信读取通过，但现有左右主手命名尚未通过独立动作标定，不能仅凭端口名称认定物理左右与关节方向正确。

从臂检查以 TCP JSON 查询成功为准：核对本地配置的左右 `host`、`port: 8080`，检查电脑网卡与机械臂的网段、网线、交换机及控制器上电状态。仅能 ping 通不能代替 `get_joint_degree` 成功返回。关节反馈按 `joint_scale: 1000.0` 换算，当前为角度值；读取静止关节不需要驱动机械臂。

### 4.5 正式采集前的判定与检查范围

正式采集前，两侧从臂和主手都应读取成功，两侧夹爪监视器应无错误，两路腕部相机应取帧成功且画面正确。采集配置保持被动模式及夹爪初始化关闭。任一项失败时先处理对应设备，不使用跳过检查或填零来替代正常反馈。

采集保存完成后，还应按[数据格式](#6-数据格式)和[数据可视化](#7-数据可视化)检查两路视频及 `observation.state`：左、右从臂夹爪分别位于索引 `6`、`13`（从 0 开始，正式 14 维顺序下）。`action` 来自主手，不能拿它替代从臂夹爪反馈。若示教中夹爪实际开合，而对应 state 全程为 0，应停止追加数据并重新排查；全程闭合时全零可以正常。

完整分项脚本对两台相机是依次各取一帧，不等于双路持续采集、编码和 30 FPS 整机性能验收。此前约 `29.81 Hz` 的真实采集验证未包含相机，不能据此承诺带两路视频后每周期均达标。

旧的 `examples/realman/check_realman_double_arm.py` 使用原生夹爪接口，其 `RealMan double-arm check passed.` **不适用于当前 RM+ 夹爪验收**；`--benchmark-samples` 也不会读取本地 YAML 中的 RM+ 配置。当前设备以本节的只读分项检查和夹爪监视器为准，历史问题与实机验证记录见[采集检查与夹爪反馈说明](realman_data_collection_check.md)。

## 5. 数据采集

先完成第 4 节的手动检查，再由外部 RealMan 系统负责主从遥操作。LeRobot 在 `passive_recording: true` 下读取主手动作、从臂反馈及相机；录制与中间重置阶段均跳过 `robot.send_action`。重置阶段仍会读取设备，但不写入 episode，也不主动复位机械臂。

模板为 `1` 个 episode、每段 `8` 秒、目标 `30 FPS`。**实际参数以本地 YAML 和命令行覆盖值为准**。为使下文的采集、续录和可视化示例使用同一数据集，下面显式指定数据集标识和目录；如已有同名目录，请换用新目录，或在确认配置一致后选择续录。

开始采集：

```bash
python -m lerobot.scripts.lerobot_record \
  --config_path=configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --dataset.repo_id=local/realman_double_arm_dataset \
  --dataset.root=datasets/realman_double_arm_dataset \
  --dataset.no_stamp=true \
  --play_sounds=false
```

如需临时修改保存目录、episode 数量或编码配置，可以在命令行覆盖：

```bash
python -m lerobot.scripts.lerobot_record \
  --config_path=configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --dataset.root=datasets/realman_double_arm_test \
  --dataset.num_episodes=3 \
  --dataset.rgb_encoder.vcodec=h264 \
  --play_sounds=false
```

续录时使用与原数据集相同的实际 `repo_id`、目录、机器人类型、FPS、相机配置和数据字段。`--resume=true` 通过 `LeRobotDataset.resume` 追加，本次 `dataset.num_episodes` 表示新增段数；续录不会再次给标识加时间后缀。若旧录制曾开启自动命名，应使用当时实际生成的带后缀标识。

例如，向同一数据集再追加 1 个 episode：

```bash
python -m lerobot.scripts.lerobot_record \
  --config_path=configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --dataset.repo_id=local/realman_double_arm_dataset \
  --dataset.root=datasets/realman_double_arm_dataset \
  --resume=true \
  --dataset.num_episodes=1 \
  --play_sounds=false
```

交互式终端支持以下录制控制；使用终端键盘后端时保持该终端聚焦。无可用显示且标准输入不是 TTY 时没有键盘监听，录制按配置的时长和段数运行。

| 按键           | 行为                             |
| -------------- | -------------------------------- |
| `n` / 右方向键 | 提前结束当前录制或重置阶段       |
| `r` / 左方向键 | 丢弃当前 episode 缓冲并重新录制  |
| `q` / `Esc`    | 结束录制流程，等待保存及清理完成 |

等待 episode 的图像写入、H.264 编码和数据保存完成后，再退出或检查文件。优先使用 `q` / `Esc` 完成结束流程；`Ctrl+C` 中断不能视为当前未保存 episode 已落盘。诊断数据、历史夹爪恒零数据与新示教数据应分目录保留，不能因字段同名就直接混合训练。

## 6. 数据格式

采集得到的数据包含：

| 字段                             | 说明                                  |
| -------------------------------- | ------------------------------------- |
| `action`                         | 14 维双主手测量状态，作为示教动作记录 |
| `observation.state`              | 14 维双从臂与夹爪的实际反馈           |
| `observation.images.left_wrist`  | 左腕 RGB 视频                         |
| `observation.images.right_wrist` | 右腕 RGB 视频                         |

正式 14 维顺序为左侧 6 个关节、左夹爪、右侧 6 个关节、右夹爪；夹爪索引为 6、13（从 0 开始）。关节与夹爪的比例来自各自配置，主手和从臂的物理映射及标尺仍需在正式示教前核对。

默认数值特征名依次为 `left_joint_1.pos` 至 `left_joint_6.pos`、`left_gripper.pos`，再到 `right_joint_1.pos` 至 `right_joint_6.pos`、`right_gripper.pos`。视频字段由左右相机的 `wrist` 配置形成；具体字段、形状与文件路径以本次数据集的 `meta/info.json` 为准。该分支写入的 Dataset 格式仍为 `v3.0`，不是 `v0.7.0`。

按 30 FPS 保存时，每个 8 秒 episode 目标约为 240 帧：

```text
8 秒 x 30 FPS = 240 帧
```

数据目录通常包含：

```text
datasets/realman_double_arm_dataset/
├── data/
├── meta/
└── videos/
```

## 7. 数据可视化

使用 LeRobot 数据集可视化工具查看视频、动作、状态和时间轴：

```bash
python -m lerobot.scripts.lerobot_dataset_viz \
  --display-mode=rerun \
  --repo-id local/realman_double_arm_dataset \
  --root datasets/realman_double_arm_dataset \
  --episode-index 0
```

如果当前机器没有图形界面，可以导出 Rerun 文件：

```bash
mkdir -p visualizations

python -m lerobot.scripts.lerobot_dataset_viz \
  --display-mode=rerun \
  --repo-id local/realman_double_arm_dataset \
  --root datasets/realman_double_arm_dataset \
  --episode-index 0 \
  --save 1 \
  --output-dir visualizations
```

生成的 `.rrd` 文件可以在有图形界面的机器上打开：

```bash
rerun visualizations/local_realman_double_arm_dataset_episode_0.rrd
```

只查看数据集基本信息：

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id local/realman_double_arm_dataset \
  --root datasets/realman_double_arm_dataset \
  --operation.type=info
```

视频实际路径以 `meta/info.json` 中的路径模板和生成文件为准。下面以左腕视频常见路径为例查看编码、尺寸和文件帧率，右腕也应检查：

```bash
ffprobe \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  datasets/realman_double_arm_dataset/videos/observation.images.left_wrist/chunk-000/file-000.mp4
```

## 8. 使用范围与相关资料

本手册中的检查入口不驱动机械臂或夹爪，也不切换末端协议。正式被动采集由外部系统完成主从遥操作，LeRobot 按 `dataset.fps` 读取数据；数据集时间戳和最新相机帧不等于传感器硬件同步。

当前默认先写入图像，在 episode 结束后进行 H.264 编码。两路图像写入和编码加入后的持续采集性能应在目标电脑上验收；修改线程数、编码方式或硬件编码后也应重新测量。没有顶部相机时保持对应配置注释。

`lerobot-record` 仅用于有主手输入的示教采集，不接受旧版 `--policy.*` 参数。模型训练入口为 `lerobot-train`，需要额外安装 `training` 及所选 policy 的依赖；第 1 节的采集安装命令不包含所有模型依赖。0.7.0 的训练跟踪使用 `trackio` 配置。

策略推理和执行入口为 `lerobot-rollout`，详见[策略部署说明](source/inference.mdx)。本分支已注册 RealMan 类型，但 RM+ 后端仍拒绝发送动作，RealMan 策略训练效果和实际闭环部署也未验收；不能直接把被动采集配置切成主动控制就用于策略执行。

| 资料                                                              | 用途                                       |
| ----------------------------------------------------------------- | ------------------------------------------ |
| [项目首页](../README.md)                                          | 项目介绍、当前能力与快速开始               |
| [夹爪修复与实机验证记录](realman_data_collection_check.md)        | 恒零原因、实机证据、设备身份及已知限制     |
| [只读检查脚本](../examples/realman/check_realman_readonly.py)     | 简洁结果及可选 JSON 报告                   |
| [夹爪连续监视器](../examples/realman/watch_realman_gripper.py)    | 连续查看 RM+ 原始位置和归一化反馈          |
| [离线采集管道检查](../examples/realman/audit_offline_pipeline.py) | 使用模拟传输验证适配与保存流程，不连接硬件 |
| [LeRobot 安装说明](source/installation.mdx)                       | 新机器环境与可选依赖安装                   |
| [LeRobot 数据集工具](source/using_dataset_tools.mdx)              | 数据集查看、编辑和管理                     |

本机厂家参考代码位于仓库同级的 `../rm_aloha_v1/`，用于查阅原始通信和遥操作实现，不随本仓库分发。相关历史验证记录中的设备参数以当时配置为准；连接前仍需核对当前本地配置。

## 9. 策略训练

先完成正式示教数据的质量检查，再使用上游训练入口。以下命令在本分支的 Conda 环境中执行，示例沿用模板的 `repo_id` 和 `root`；如果已修改本地配置，请同步替换这些值。

```bash
# 安装训练依赖，同时保留采集与相机依赖
python -m pip install -e ".[core_scripts,intelrealsense,training]"

lerobot-train \
  --dataset.repo_id=local/realman_double_arm_dataset \
  --dataset.root=datasets/realman_double_arm_dataset \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/realman_act \
  --job_name=realman_act \
  --batch_size=8 \
  --steps=10000 \
  --trackio.enable=false
```

示例假定可用的 CUDA GPU，输出目录应尚不存在；批量大小与训练步数按数据规模和显存调整。ACT 默认视觉骨干可能需要下载预训练权重。选择 Diffusion、SmolVLA 或 π 系列时，另装对应扩展，例如 `python -m pip install -e ".[diffusion]"`、`python -m pip install -e ".[smolvla]"` 或 `python -m pip install -e ".[pi]"`。

模型配置见 [ACT](source/policy_act_README.md)、[Diffusion](source/policy_diffusion_README.md) 和 [SmolVLA](source/policy_smolvla_README.md)。RealMan 任务的训练效果与闭环部署尚未验收，当前 RM+ 适配仍拒绝发送动作。

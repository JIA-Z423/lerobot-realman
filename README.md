# RealMan 双臂 LeRobot 项目说明

本文档说明如何使用 RealMan 双臂 ALOHA 风格平台接入 [LeRobot](https://github.com/huggingface/lerobot)，完成被动式数据采集。

本仓库主体基于 Hugging Face LeRobot 项目，当前工作是在 LeRobot 的机器人、主手、相机和数据集接口之上，增加本实验室 RealMan 双臂设备的硬件适配和采集流程。后续模型训练与部署推理也将继续沿用 LeRobot 的数据集、策略训练和真实机器人推理接口。

当前已完成硬件采集流程，能够记录：

- 双主手动作
- 双从臂状态
- 左右腕部相机视频

默认使用被动采集模式。机械臂由外部 RealMan 或厂家控制系统驱动，LeRobot 只负责同步记录数据，不主动下发机械臂控制命令。

## 1. 项目说明

上游项目：

- LeRobot: <https://github.com/huggingface/lerobot>
- LeRobot 文档: <https://huggingface.co/docs/lerobot>

本项目中的 `rm_aloha_v1/` 为睿尔曼公司提供的双臂配套项目代码，主要用于参考厂家原始通信方式、主手协议和 RealMan ALOHA 使用流程。

当前 LeRobot 集成代码主要包括：

| 路径 | 说明 |
| --- | --- |
| `configs/hardware/realman_double_arm_hardware_config.yaml` | 双臂采集配置文件 |
| `examples/realman/check_realman_double_arm.py` | 双臂硬件检查脚本 |
| `src/lerobot/robots/realman_follower/` | 单个 RealMan 从臂封装 |
| `src/lerobot/robots/double_realman_follower/` | RealMan 双从臂封装 |
| `src/lerobot/teleoperators/rm_aloha_master/` | 单个 RM ALOHA 主手封装 |
| `src/lerobot/teleoperators/double_rm_aloha_master/` | 双主手封装 |

## 1.1 项目目录

后续使用重点关注以下目录：

| 路径 | 说明 |
| --- | --- |
| `src/lerobot/` | LeRobot 主体代码和 RealMan 硬件适配代码 |
| `configs/hardware/` | 硬件采集配置 |
| `examples/realman/` | RealMan 双臂检查脚本 |
| `datasets/` | 本地采集数据，默认不提交 |
| `outputs/` | 训练、评估和运行输出，默认不提交 |
| `visualizations/` | Rerun 可视化导出文件，默认不提交 |
| `rm_aloha_v1/` | 睿尔曼公司提供的配套项目代码 |


## 1.2 版本管理说明

建议将本仓库作为独立 Git 仓库维护：

```bash
git remote add origin git@github.com:<your_github_username>/<your_repo>.git
git remote add upstream https://github.com/huggingface/lerobot.git
```



本仓库的公开配置使用占位符。真实设备配置请放在 `configs/hardware/realman_double_arm_hardware_config.local.yaml`。

## 2. 环境安装

进入 LeRobot 环境：

```bash
conda activate lerobot
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

检查主要依赖：

```bash
python -c "import serial, pyrealsense2, draccus; print('ok')"
```

如果需要从零安装依赖：

参考LeRobot 安装文档：<https://huggingface.co/docs/lerobot/installation>

## 3. 硬件参数

公开配置使用以下占位符。运行前请在本地 `.local.yaml` 配置中替换为真实设备参数：

| 设备 | 参数 |
| --- | --- |
| 左从臂 | `REPLACE_WITH_LEFT_REALMAN_ARM_HOST:8080` |
| 右从臂 | `REPLACE_WITH_RIGHT_REALMAN_ARM_HOST:8080` |
| 左主手 | `REPLACE_WITH_LEFT_MASTER_SERIAL_PORT` |
| 右主手 | `REPLACE_WITH_RIGHT_MASTER_SERIAL_PORT` |
| 左腕相机 | `REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL` |
| 右腕相机 | `REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL` |

如果重启后串口名发生变化，建议把 `/dev/ttyUSB*` 改成稳定的设备路径：

```bash
ls -l /dev/serial/by-id/
```

## 4. 配置文件

公开模板配置文件：

```bash
configs/hardware/realman_double_arm_hardware_config.yaml
```

真实采集建议使用本地配置文件：

```bash
configs/hardware/realman_double_arm_hardware_config.local.yaml
```

关键默认项：

```yaml
passive_recording: true

dataset:
  repo_id: local/realman_double_arm_dataset
  root: datasets/realman_double_arm_dataset
  num_episodes: 1
  fps: 10
  episode_time_s: 8
  reset_time_s: 0
  streaming_encoding: false
  vcodec: h264
  push_to_hub: false

robot:
  type: double_realman_follower

teleop:
  type: double_rm_aloha_master
```

正式采集前通常需要确认：

- `dataset.repo_id`
- `dataset.root`
- `dataset.single_task`
- `dataset.num_episodes`
- 相机序列号
- 主手串口路径

当前双臂夹爪默认启用，夹爪硬件行程按 `gripper_scale=1000.0` 缩放为 `0~1` 后写入动作和观测。

## 5. 硬件检查

运行完整检查：

```bash
python examples/realman/check_realman_double_arm.py \
  --left-host=<left_arm_host> \
  --right-host=<right_arm_host> \
  --left-master-port=<left_master_serial_port> \
  --right-master-port=<right_master_serial_port> \
  --left-wrist-serial=<left_wrist_camera_serial> \
  --right-wrist-serial=<right_wrist_camera_serial>
```

正常输出应包含：

```text
RealMan double-arm check passed.
```

测试只读循环速度：

```bash
python examples/realman/check_realman_double_arm.py \
  --skip-followers --skip-masters --skip-cameras \
  --benchmark-samples=50
```

其他常用检查：

```bash
python examples/realman/check_realman_double_arm.py --skip-cameras
python examples/realman/check_realman_double_arm.py --check-gripper
```

## 6. 数据采集

默认配置为本地采集 `1` 个 episode，每个 episode `8` 秒，`10 FPS`，数据保存到：

```bash
datasets/realman_double_arm_dataset
```

开始采集：

```bash
python -m lerobot.scripts.lerobot_record \
  --config_path=configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --play_sounds=false
```

如需临时修改保存目录或 episode 数量，可以在命令行覆盖：

```bash
python -m lerobot.scripts.lerobot_record \
  --config_path=configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --dataset.root=datasets/realman_double_arm_test \
  --dataset.num_episodes=3 \
  --play_sounds=false
```

继续采集已有本地数据集：

```bash
python -m lerobot.scripts.lerobot_record \
  --config_path=configs/hardware/realman_double_arm_hardware_config.local.yaml \
  --resume=true \
  --play_sounds=false
```

## 7. 数据格式

采集得到的数据包含：

| 字段 | 说明 |
| --- | --- |
| `action` | 14 维双主手动作 |
| `observation.state` | 14 维双从臂状态 |
| `observation.images.left_wrist` | 左腕 RGB 视频 |
| `observation.images.right_wrist` | 右腕 RGB 视频 |

默认配置下，每个 8 秒 episode 约为 80 帧：

```text
8 秒 x 10 FPS = 80 帧
```

数据目录通常包含：

```text
datasets/realman_double_arm_dataset/
├── data/
├── meta/
└── videos/
```

## 8. 数据可视化

使用 LeRobot 数据集可视化工具查看视频、动作、状态和时间轴：

```bash
lerobot-dataset-viz \
  --repo-id local/realman_double_arm_dataset \
  --root datasets/realman_double_arm_dataset \
  --episode-index 0
```

如果当前机器没有图形界面，可以导出 Rerun 文件：

```bash
mkdir -p visualizations

lerobot-dataset-viz \
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
lerobot-edit-dataset \
  --repo_id local/realman_double_arm_dataset \
  --root datasets/realman_double_arm_dataset \
  --operation.type=info
```

快速查看视频元信息：

```bash
conda run -n lerobot ffprobe \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  datasets/realman_double_arm_dataset/videos/observation.images.left_wrist/chunk-000/file-000.mp4
```

## 9. 模型训练

待补充。


- 数据集路径和 `repo_id` 选择
- ACT / Diffusion / SmolVLA 等策略训练入口
- 训练参数配置
- checkpoint 保存位置
- 训练日志和结果查看方式

## 10. 部署推理

待补充。



- 加载训练好的 policy checkpoint
- 使用 RealMan 双臂配置进行实机推理
- 推理前硬件检查
- 安全限位和急停注意事项
- 评估 episode 的保存与回放

## 11. 说明

- 当前版本仅完成双臂、两路 `640x480` 腕部相机的`10 FPS`数据采集。
- 机械臂和主手的只读循环速度可以超过 `30 Hz`。
- 默认关闭实时视频编码，先写入图片，episode 结束后再编码，采集帧率更稳定。
- `30 FPS` 或实时编码的主要瓶颈通常在图像写入或视频编码。
- GPU 或硬件编码可能提升采集速度，但需要在目标电脑上实际测试。
- 顶视或外部相机默认未启用，确认型号和序列号后再加入配置。
- 无显示器环境下，`pynput` 或 `DISPLAY` 相关警告不影响被动采集。

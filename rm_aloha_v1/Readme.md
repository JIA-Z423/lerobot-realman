# RM_Aloha_1.0
> 此为睿尔曼公司提供的配套项目代码，RM_Aloha_1.0版本代码，主要包括RM_Aloha_1.0版本的代码和文档。   

## 1. 代码结构
```
RM_Aloha_1.0
├── rm_aloha_v1
│   ├── aloha_data  # 存放数据集
│   ├── aloha_data_visualization    # 数据集可视化
│   ├── aloha_model_ckpt    # 模型保存
│   ├── network   # 网络模型
│   │   ├── detr
│   │   ├── robomimic
│   │   ├── policy.py
│   ├── RM_robotic  # 机器人控制
│   ├── src  # 主要代码
│   │   ├── constants.py
│   │   ├── ...
│   ├── utils   # 工具
│   ├── record_episodes.py # 采集数据集
│   ├── imitate_episodes.py   # 训练数据集
│   ├── read_hdf5.py # 可视化数据集
```

## 2. 环境配置
环境配置如下([官网](https://github.com/tonyzhaozh/act))：
```shell
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
```

## 3. 数据集采集
```shell
python record_episodes.py
```
## 4. 数据集可视化
```shell    
python read_hdf5.py
```

## 5. 数据集训练
```shell
python imitate_episodes.py --task_name aloha --ckpt_dir aloha_model_ckpt
```

## 6. 模型测试
```shell
python imitate_episodes.py --task_name aloha --ckpt_dir aloha_model_ckpt --eval
```



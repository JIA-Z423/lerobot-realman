import pyrealsense2 as rs
import numpy as np
import cv2 as cv

# ctx = rs.context()
# print(ctx.devices[0], '\n', ctx.devices[1], '\n', ctx.devices[2])


ctx = rs.context()
device_count = len(ctx.devices)

print(f"检测到 {device_count} 个 RealSense 设备\n")

for i in range(device_count):
    device = ctx.devices[i]
    print(f"设备 {i}: {device}")
    print(f"  名称: {device.get_info(rs.camera_info.name)}")
    print(f"  序列号: {device.get_info(rs.camera_info.serial_number)}")
    print(f"  固件版本: {device.get_info(rs.camera_info.firmware_version)}")
    print()

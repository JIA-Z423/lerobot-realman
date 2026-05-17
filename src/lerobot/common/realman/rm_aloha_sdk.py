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

from __future__ import annotations

import ctypes
import platform
from pathlib import Path


RM65 = 65


ERROR_MESSAGES = {
    1: "CONTROLLER_DATA_RETURN_FALSE",
    2: "INIT_MODE_ERR",
    3: "INIT_TIME_ERR",
    4: "INIT_SOCKET_ERR",
    5: "SOCKET_CONNECT_ERR",
    6: "SOCKET_SEND_ERR",
    7: "SOCKET_TIME_OUT",
    8: "UNKNOWN_ERR",
    9: "CONTROLLER_DATA_LOSE_ERR",
    10: "CONTROLLER_DATE_ARR_NUM_ERR",
    11: "WRONG_DATA_TYPE",
    12: "MODEL_TYPE_ERR",
    13: "CALLBACK_NOT_FIND",
    14: "ARM_ABNORMAL_STOP",
    18: "CONTROLLER_BUSY",
    19: "ILLEGAL_INPUT",
    20: "QUEUE_LENGTH_FULL",
}


def _dll_path() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(__file__).resolve().parent / "RM_Base.dll"
    if system == "Linux":
        return Path(__file__).resolve().parent / "lib" / "libRM_Base.so"
    raise RuntimeError(f"Unsupported RealMan SDK platform: {system}")


class GripperState(ctypes.Structure):
    _fields_ = [
        ("enable_state", ctypes.c_bool),
        ("status", ctypes.c_int),
        ("error", ctypes.c_int),
        ("mode", ctypes.c_int),
        ("current_force", ctypes.c_int),
        ("temperature", ctypes.c_int),
        ("actpos", ctypes.c_int),
    ]


class Arm:
    """Minimal wrapper for the RM_Aloha SDK used by LeRobot.

    The original vendor project ships a very broad ctypes wrapper. LeRobot only
    needs connection, joint position, CANFD joint command, and gripper route /
    position / state calls, so this class keeps the supported surface explicit.
    """

    def __init__(self, dev_mode: int, ip: str, port: int = 8080):
        self.code = dev_mode
        while self.code >= 10:
            self.code //= 10

        self._dll = ctypes.cdll.LoadLibrary(str(_dll_path()))
        self._configure_signatures()
        self._dll.RM_API_Init(dev_mode, 0)
        self.nSocket = self._dll.Arm_Socket_Start(ip.encode("gbk"), port, 200)

    def _configure_signatures(self) -> None:
        self._dll.RM_API_Init.argtypes = (ctypes.c_int, ctypes.c_void_p)
        self._dll.RM_API_Init.restype = ctypes.c_int
        self._dll.RM_API_UnInit.restype = ctypes.c_int

        self._dll.Arm_Socket_Start.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
        self._dll.Arm_Socket_Start.restype = ctypes.c_int
        self._dll.Arm_Socket_State.argtypes = (ctypes.c_int,)
        self._dll.Arm_Socket_State.restype = ctypes.c_int
        self._dll.Arm_Socket_Close.argtypes = (ctypes.c_int,)

        self._dll.Get_Joint_Degree.argtypes = (ctypes.c_int, ctypes.c_float * 7)
        self._dll.Get_Joint_Degree.restype = ctypes.c_int

        self._dll.Set_Gripper_Route.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
        )
        self._dll.Set_Gripper_Route.restype = ctypes.c_int
        self._dll.Set_Gripper_Position.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int,
        )
        self._dll.Set_Gripper_Position.restype = ctypes.c_int
        self._dll.Get_Gripper_State.argtypes = (ctypes.c_int, ctypes.POINTER(GripperState))
        self._dll.Get_Gripper_State.restype = ctypes.c_int

    def Arm_Socket_State(self):
        return self._format_result(self._dll.Arm_Socket_State(self.nSocket))

    def Arm_Socket_Close(self) -> None:
        self._dll.Arm_Socket_Close(self.nSocket)

    def RM_API_UnInit(self):
        return self._format_result(self._dll.RM_API_UnInit())

    def Get_Joint_Degree(self):
        joint = (ctypes.c_float * 7)()
        result = self._dll.Get_Joint_Degree(self.nSocket, joint)
        return self._format_result(result), list(joint)

    def Movej_CANFD(self, joint, follow, expand=0):
        if self.code == 6:
            joints = (ctypes.c_float * 6)(*joint)
            self._dll.Movej_CANFD.argtypes = (
                ctypes.c_int,
                ctypes.c_float * 6,
                ctypes.c_bool,
                ctypes.c_int,
            )
        else:
            joints = (ctypes.c_float * 7)(*joint)
            self._dll.Movej_CANFD.argtypes = (
                ctypes.c_int,
                ctypes.c_float * 7,
                ctypes.c_bool,
                ctypes.c_int,
            )
        self._dll.Movej_CANFD.restype = ctypes.c_int
        return self._format_result(self._dll.Movej_CANFD(self.nSocket, joints, follow, expand))

    def Set_Gripper_Route(self, min_limit, max_limit, block=True):
        result = self._dll.Set_Gripper_Route(self.nSocket, min_limit, max_limit, block)
        return self._format_result(result)

    def Set_Gripper_Position(self, position, block=True, timeout=30):
        result = self._dll.Set_Gripper_Position(self.nSocket, position, block, timeout)
        return self._format_result(result)

    def Get_Gripper_State(self):
        state = GripperState()
        result = self._dll.Get_Gripper_State(self.nSocket, ctypes.byref(state))
        return self._format_result(result), state

    @staticmethod
    def _format_result(code: int):
        if code == 0:
            return 0
        return f"{code}: {ERROR_MESSAGES.get(code, 'UNKNOWN_ERR')}"

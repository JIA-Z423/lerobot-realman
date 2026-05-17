#!/usr/bin/env python
# -*- coding: utf-8 -*-
import serial
import binascii
import time
import numpy as np


def arm_conf():
    l_port = "REPLACE_WITH_LEFT_MASTER_SERIAL_PORT"
    r_port = "REPLACE_WITH_RIGHT_MASTER_SERIAL_PORT"
    baudrate = 460800
    hex_data = "55 AA 02 00 00 67"
    #

    l_ser = serial.Serial(l_port, baudrate, timeout=0)
    r_ser = serial.Serial(r_port, baudrate, timeout=0)

    bytes_to_send = binascii.unhexlify(hex_data.replace(" ", ""))
    l_ser.write(bytes_to_send)
    r_ser.write(bytes_to_send)
    time.sleep(1)
    return l_ser, r_ser, bytes_to_send


def bytes_to_signed_int(byte_data):
    value = int.from_bytes(byte_data, byteorder='little', signed=True)
    return value


def get_Joint(hex_received):
    J1 = hex_received[14:22]
    J1_byte_data = bytearray.fromhex(J1)
    Joint1 = bytes_to_signed_int(J1_byte_data) / 10000.0

    J2 = hex_received[24:32]
    J2_byte_data = bytearray.fromhex(J2)
    Joint2 = bytes_to_signed_int(J2_byte_data) / 10000.0

    J3 = hex_received[34:42]
    J3_byte_data = bytearray.fromhex(J3)
    Joint3 = bytes_to_signed_int(J3_byte_data) / 10000.0

    J4 = hex_received[44:52]
    J4_byte_data = bytearray.fromhex(J4)
    Joint4 = bytes_to_signed_int(J4_byte_data) / 10000.0

    J5 = hex_received[54:62]
    J5_byte_data = bytearray.fromhex(J5)
    Joint5 = bytes_to_signed_int(J5_byte_data) / 10000.0

    J6 = hex_received[64:72]
    J6_byte_data = bytearray.fromhex(J6)
    Joint6 = bytes_to_signed_int(J6_byte_data) / 10000.0

    G7 = hex_received[74:82]
    G7_byte_data = bytearray.fromhex(G7)
    Grasp = bytes_to_signed_int(G7_byte_data)

    return Joint1, Joint2, Joint3, Joint4, Joint5, Joint6, Grasp


def get_master_act(l_ser, r_ser, byte_send):
    l_ser.write(byte_send)
    r_ser.write(byte_send)
    l_bytes_received = l_ser.read(l_ser.inWaiting())
    r_bytes_received = r_ser.read(r_ser.inWaiting())
    l_hex_received = binascii.hexlify(l_bytes_received).decode('utf-8').upper()
    r_hex_received = binascii.hexlify(r_bytes_received).decode('utf-8').upper()
    l_actions = get_Joint(l_hex_received)
    r_actions = get_Joint(r_hex_received)
    return np.array((l_actions + r_actions))


if __name__ == '__main__':
    # print(get_master_act().shape)
    l_ser, r_ser, byte_send = arm_conf()
    print(l_ser, '\n', r_ser)
    for i in range(10):
        time1 = time.time()
        pose = get_master_act(l_ser, r_ser, byte_send)
        time2 = time.time()
        print("time:", time2 - time1)
        print(pose)
        time.sleep(0.5)

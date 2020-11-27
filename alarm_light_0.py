import serial
import time

ser = serial.Serial('COM8', 9600, timeout=0.5)                        # 设置一个串口对象(继电器)

def alarm_0(light_state):
    if light_state == 1:
        signal = bytes([0XFE,0X05,0X00,0X00,0XFF,0X00,0X98,0X35])       # 打开继电器，亮灯
        ser.write(signal)
    if light_state == 0:
        signal = bytes([0XFE,0X05,0X00,0X00,0X00,0X00,0XD9,0XC5])       # 打开继电器，灭灯
        ser.write(signal)

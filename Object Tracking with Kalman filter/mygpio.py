# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import RPi.GPIO as GPIO
import time

def blink():
    
    GPIO.cleanup()
    GPIO.setmode(GPIO.BOARD)
    pin=7
    GPIO.setup(pin,GPIO.OUT,initial=0)
    GPIO.output(pin,1)
    time.sleep(.01)
    GPIO.output(pin,0)
    time.sleep(.01)
    GPIO.cleanup()












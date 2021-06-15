#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:04:57 2018

@author: pi
"""

import cv2
import time
from stream_thread_pi import PiVideoStream

vs = PiVideoStream(rotation=180,b=60,c=0).start()
time.sleep(2.0)
 
# loop over some frames...this time using the threaded stream
while (1):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
#    frame = imutils.resize(frame, width=400)
 
    # check to see if the frame should be displayed to our screen
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
    # update the FPS counter
 
# stop the timer and display FPS information
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

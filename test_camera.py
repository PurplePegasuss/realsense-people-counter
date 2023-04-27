import pyrealsense2 as rs
import numpy as np
import cv2
import os


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 15)

cfg = pipeline.start(config)
dev = cfg.get_device()

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0)

depth_sensor = dev.first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 2)

pMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=16, 
            detectShadows=True)

def processMog(frame):
    # frame = self.cam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = pMOG2.apply(gray, learningRate=-1)

    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    '''mask = cv2.dilate(mask, self.kernelDilate)
    mask = cv2.erode(mask, self.kernelErode)
    mask = cv2.dilate(mask, self.kernelDilate)'''
    mask = cv2.dilate(mask, None, iterations=5)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=5)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, None, 2)

    return frame, mask

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir1_frame = frames.get_infrared_frame(1)
        if not depth_frame or not color_frame or not ir1_frame:
             continue
        
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ir1_image = np.asanyarray(ir1_frame.get_data())

        frame, mask = processMog(color_image)
        contours0, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contout_image = cv2.drawContours(frame, contours0, -1, (0,255,0), 2)

        # cv2.namedWindow('RGB_RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('Depth_RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('IR_RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RGB_RealSense', color_image)
        # cv2.imshow('Depth_Realsense', depth_image)
        # cv2.imshow('IR_RealSense', ir1_image)
        # combined_window = np.hstack((contout_image, mask))
        # combined_window = np.hstack((combined_window, ir1_image))
        cv2.imshow('Output', contout_image)
        cv2.imshow('Mask', mask)
        # cv2.imshow('Points', ir1_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
finally:
    pipeline.stop()

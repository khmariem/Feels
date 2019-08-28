import pyrealsense2 as s
import numpy.ma as ma
import numpy as np
import cv2 as cv
from copy import copy

def paint(img_path=None):

    if img_path==None:
        img = np.zeros([480,640,3],dtype=np.uint8)
        img.fill(255)
    else:
        img = cv.imread(img_path,1)
        img = cv.resize(img,  (640,480) , interpolation = cv.INTER_AREA) 

    pipeline = s.pipeline()

    config = s.config()
    config.enable_stream(s.stream.depth, 640, 360, s.format.z16, 30)
    config.enable_stream(s.stream.color, 640, 480, s.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor=profile.get_device().first_depth_sensor()
    depth_scale=depth_sensor.get_depth_scale()

    align=s.align(s.stream.color)

    try:
        while True:
            img_temp=img.copy()
            frames=pipeline.wait_for_frames()
            aligned_frames=align.process(frames)

            depth = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(depth.get_data())

            color = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color.get_data())

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth_image,0))
            mask_depth[:40,:]=0
            mask_depth[-40:,:]=0
            mask_depth[:,:40]=0
            mask_depth[:,-40:]=0

            val1=np.sum(mask_depth,axis=0)
            #ind1=list(np.where(val1==np.amax(val1))[0])
            ind1=list(np.where(val1>250)[0])

            val2=np.sum(mask_depth,axis=1)
            ind2=list(np.where(val2>250)[0])


            if ind1!=None and ind2!=None:
                try:
                    ind1=int(np.mean(ind1))
                    ind2=int(np.mean(ind2))
                    #img_temp[ind2-15:ind2+15,ind1-15:ind1+15,:]=0
                    cv.circle(img,(ind1,ind2), 10, (0,0,255), -1)
                except:
                    pass
            #apply mirror transformation to original image
            color_image=cv.flip(color_image, 1)
            # Stack both images horizontally
            images = np.hstack((color_image, img_temp))
            cv.imshow('image',images)
            cv.waitKey(1)

    finally:
        pipeline.stop()

if __name__=="__main__":
    paint()

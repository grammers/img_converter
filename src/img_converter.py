#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import h5py 
from torchvision.transforms import transforms 
import torch.utils.data as data
import numpy as np 
import torch
import random

class FrameDataset(data.Dataset):
    
    def __init__(self, f, transform=None, test = False):
        self.f = f 
        self.transform = transform 
        self.test = test
        
    def __getitem__(self, index):

        rgb = np.array(self.f["rgb"][index])
        label = np.array((self.f["labels"][index] - self.f["Mean"])) ## same mean through train, test 
        
        t_rgb = torch.zeros(rgb.shape[0], 3, 224, 224)
        
        prob = random.uniform(0, 1)

        if self.transform is not None:
            for i in range(rgb.shape[0]):
                if (prob > 0.5 and not self.test):
                    flip_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(1.0)])
                    #rgb[i,:,:,:] = flip_transform(rgb[i,:,:,:])
                t_rgb[i,:,:,:] = self.transform(rgb[i,:,:,:])

                
        return rgb, t_rgb, label
    
    def __len__(self):
        return len(self.f["rgb"])
        
class image_converter:

    def __init__(self):
        
        self.hfp_test = h5py.File('/home/hexa/catkin_workspaces/catkin_samuel/src/nearCollision/data/h5_files/6ImageTest.h5', 'r')
        self.mean = self.hfp_test["Mean"][()]
        self.var = self.hfp_test["Variance"][()]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.test_loader = data.DataLoader(FrameDataset(f = self.hfp_test, transform = transforms.Compose([transforms.ToTensor(), self.normalize]), test = True), batch_size=1)
        
        
        self.bridge = CvBridge()
        
        self.image_pub = rospy.Publisher("/usb_cam/image_raw", Image, queue_size = 10)
        #self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

    def callback(self):
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.image_pub.publish(cv_image)
        '''
        #(rows,cols,channels) = cv_image.shape
        #if cols > 60 and rows > 60 :
        #    cv2.circle(cv_image, (50,50), 10, 255)

        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)
        rate = rospy.Rate(1)
        for iter, (img, rgb, label) in enumerate(self.test_loader,0):
            if rospy.is_shutdown():
                break

            (channels, rows, cols) = img[0,-1,:,:,:].shape
            cv_img = (rows, cols, channels)
            
            try:
                self.image_pub.publish(
                self.bridge.cv2_to_imgmsg(
                img[0,-1,:,:,:].numpy(), "bgr8"))
            except CvBridgeError as e:
                print(e)
            print("loop")
            rate.sleep()

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        ic.callback()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

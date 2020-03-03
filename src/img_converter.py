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

nr_img = 1
# used to extrakt data out of .h5 file
class FrameDataset(data.Dataset):
    
    def __init__(self, f, transform=None):
        self.f = f 
        self.transform = transform 
        
    def __getitem__(self, index):

        rgb = np.array(self.f["rgb"][index])
        label = np.array((self.f["labels"][index] - self.f["Mean"])) ## same mean through train, test 
        
        t_rgb = torch.zeros(rgb.shape[0], 3, 224, 224)
        
        prob = random.uniform(0, 1)

        if self.transform is not None:
            if nr_img == 6:
                for i in range(rgb.shape[0]):
                    t_rgb[i,:,:,:] = self.transform(rgb[i,:,:,:])
            elif nr_img == 1:
                t_rgb[:,:,:] = self.transform(rgb[:,:,:])

                
        return rgb, t_rgb, label
    
    def __len__(self):
        return len(self.f["rgb"])
        
class image_converter:

    def __init__(self):
        
        if nr_img == 6:
            self.hfp_test = h5py.File(
            '/home/hexa/catkin_workspaces/catkin_samuel/src/nearCollision/data/h5_files/6ImageTest.h5', 'r')
        elif nr_img == 1:
            self.hfp_test = h5py.File(
            '/home/hexa/catkin_workspaces/catkin_samuel/src/nearCollision/data/h5_files/SingleImageTest.h5', 'r')
        
        self.mean = self.hfp_test["Mean"][()]
        self.var = self.hfp_test["Variance"][()]
        self.normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.test_loader = data.DataLoader(
        FrameDataset(f = self.hfp_test, transform = transforms.Compose(
        [transforms.ToTensor(), self.normalize])), batch_size=1)
        
        
        self.bridge = CvBridge()
        
        self.image_pub = rospy.Publisher("/usb_cam/image_raw", Image, queue_size = 10)

    def spin(self):
        if nr_img == 6:
            rate = rospy.Rate(1)
        elif nr_img == 1:
            rate = rospy.Rate(4)
        #iterate over the mimages in the .h5 file
        for iter, (img, rgb, label) in enumerate(self.test_loader,0):
            if rospy.is_shutdown():
                break
            
            #convert cv img format to ros img format
            #and publish
            try:
                if nr_img == 6:
                    self.image_pub.publish(
                        self.bridge.cv2_to_imgmsg(
                        img[0,-1,:,:,:].numpy(), "bgr8"))
                elif nr_img == 1:
                    self.image_pub.publish(
                        self.bridge.cv2_to_imgmsg(
                        img[-1,:,:,:].numpy(), "bgr8"))
                    
            except CvBridgeError as e:
                print(e)
            print("loop")
            # sleep to give cnn node time to work
            rate.sleep()

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        ic.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

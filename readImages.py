

import cv2
import mediapipe as mp

import numpy as np
from os import listdir
from os.path import isfile, join
import time
import matplotlib.pyplot as plt
import os
import shutil




# get the path/directory
folder_dir = r"C:\Users\aryav\Desktop\Github\Physion\images"
for images in os.listdir(folder_dir):
	if (images.endswith(".jpg")):
		print(images)

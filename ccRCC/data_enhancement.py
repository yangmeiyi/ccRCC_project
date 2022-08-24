from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imshow, imsave
import torch
import numpy as np
import time


import os
import random
import shutil


def cover_files(source_dir, target_ir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, target_ir)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def moveFile(file_dir, save_dir):
    ensure_dir_exists(save_dir)
    path_dir = os.listdir(file_dir)  # Image original path
    filenumber = len(path_dir)
    rate = 0.5 
    picknumber = int(filenumber * rate)  
    sample = random.sample(path_dir, picknumber)  
    for name in sample:
        path = file_dir + name
        image = plt.imread(path)
        if rate == 1:
            imsave(save_dir + name, image)
        else:
            corrupted_image = corrupt(image, corruption_name='gaussian_noise', severity=1)
            imsave(save_dir + name, corrupted_image)



def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  
        os.makedirs(path) 
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


if __name__ == '__main__':

    path = "/home/yangmy/MedTData/dataClean/cleaned/train/1/"
    path_save = "/home/yangmy/MedTData/dataClean/cleaned/Blur_train/1/"
    dirs = os.listdir(path)
    for file in dirs:
        file_dir = path + file + '/' 
        print(file_dir)
        save_dir = path_save + file  
        print(save_dir)
        mkdir(save_dir)  
        save_dir = save_dir + '/'
        moveFile(file_dir, save_dir)



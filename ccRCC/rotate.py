import numpy as np
from PIL import Image
from scipy.misc import imshow, imsave
import torchvision
import os


file = "./3_10139937_IMG-0002-00016.jpg_R_256_orig.bmp"
img = Image.open(file)
img2 = img.rotate(180)
# img3 = img.RandomCrop
brightness = 0
contrast = 0
saturation = 1
hue = 0.3
RC = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
rc_image = RC(img)
HF = torchvision.transforms.RandomHorizontalFlip()
hf_image = HF(img)

imsave("/home/yangmy/Code/VisualTrans/color_jitter.jpg", rc_image)

import io
import torch
import requests
from PIL import Image
from torchvision import transforms
# import models.tinyImage as models
from pre_models.TransResNet import Base_TransRseVit
from pre_models.Transdensenet import Base_TransDensenet
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import os
import random
import torch.backends.cudnn as cudnn
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
use_cuda = torch.cuda.is_available()
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)


model = Base_TransRseVit()
net = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(r"/home/yangmy/Code/VisualTrans/kidney_save_model/TransResNet_best.pth.tar")
net.load_state_dict(checkpoint['state_dict'])
finalconv_name = 'transformer'
net.eval()


features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net.module._modules.get(finalconv_name).register_forward_hook(hook_feature)
params = list(net.parameters())
weight_softmax = np.array(np.squeeze(params[-2].data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (512, 512)
    nc, h, w = feature_conv.shape  # 1 9 128
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((w, h))) # 128 9
        print("h:", h)
        cam = cam.reshape(3, 3)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam



preprocess = transforms.Compose([
   transforms.Resize((94,94)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def Hot_map(path, original_path, Resnet_path, Transformer_path):
    list_path = os.listdir(path)
    for file in list_path:
        person_path = os.path.join(path, file)
        list_image = os.listdir(person_path)
        for imag_name in list_image:
            image_path = os.path.join(person_path, imag_name) # '/home/yangmy/MedTData/cleaned_tumor/test/1/10027174/3_10027174_IMG-0002-00025.jpg_L_256.bmp'
            img_pil = Image.open(image_path)

            img_tensor = preprocess(img_pil)
            img_variable = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False)
            preds = net(img_variable)
            h_x = F.softmax(preds, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            save_Transformer_path = os.path.join(Transformer_path, imag_name)
            cv2.imwrite(save_Transformer_path, result)






if __name__ == '__main__':

    path_0 = "/home/yangmy/MedTData/cleaned_tumor/test/0/"
    path_1 = "/home/yangmy/MedTData/cleaned_tumor/test/1/"
    original_path = "/home/yangmy/MedTData/hotmap_image_20211129/original/"
    Resnet_path = "/home/yangmy/MedTData/hotmap_image_20211129/Resnet/"
    Transformer_path = "/home/yangmy/MedTData/hotmap_image_20211129/Transoformer/"
    Hot_map(path_1, original_path, Resnet_path, Transformer_path)



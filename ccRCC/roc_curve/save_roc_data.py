# Import Libraries
from __future__ import print_function
import PIL
import time
import torch
import torchvision
import torch.nn.parallel
import os
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from TransRestNet_model.py_identity import Identity

from pre_models.Transdensenet import Base_TransDensenet
from pre_models.TransRegnet import Base_TransRegnet
from pre_models.TransSENet import Base_TransSENet
from pre_models.Cait import Base_Cait
from pre_models.Vit import Base_ViT
from pre_models.TransResNet import Base_TransRseVit
from pre_models.TransInception import Base_TransInception_v3
import torch.nn.parallel
from sklearn import metrics
from TransRestNet_model.utils import Bar, Logger, AverageMeter, accuracy
from dataloader import CCRFolder
import pandas as pd
import shutil
from matplotlib import rcParams


import numpy as np
from sklearn import metrics
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

use_cuda = torch.cuda.is_available()

# Training settings
lr = 0.0001
gamma = 0.7
N_EPOCHS = 200
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100

pre = []
label = []

config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)


def get_RoC(df):
    label = df['labels']
    scores = df['neg_preds']
    fpr, tpr, _ = metrics.roc_curve(label, scores)
    np.save('SENet_fpr.npy', fpr)
    np.save('SENet_tpr.npy', tpr)
    print(len(fpr), len(fpr))
    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    # plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    axes.spines['bottom'].set_linewidth(2);
    axes.spines['left'].set_linewidth(2);
    axes.spines['right'].set_linewidth(2);
    axes.spines['top'].set_linewidth(2);

    x = [0, 1]
    y = [0, 1]
    plt.xlabel('1-Specificity', fontsize=20)
    plt.ylabel('Sensitivity', fontsize=20)

    x_tricks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.xticks(x_tricks, fontproperties={'family': 'Times New Roman', "size": 20})
    plt.yticks(x_tricks, fontproperties={'family': 'Times New Roman', "size": 20}, )

    plt.tick_params(axis='x', width=2, direction='out', labelsize=18)
    plt.tick_params(axis='y', width=2, direction='out', labelsize=18)
    plt.plot(fpr, tpr, linewidth=4, color='r', label='TransResNet(area = 0.92)')
    # plt.fill_between(fpr, tpr, interpolate=True, color='#FF9492', alpha=0.5)
    plt.legend(loc='lower right', fontsize=20)


    plt.plot(x, y, linewidth=4, color='gray', linestyle='--')
    plt.grid()
    plt.savefig("./roc.pdf", dpi=1000)
    plt.show()
    auc_resnet = metrics.auc(fpr, tpr)
    print(auc_resnet)
    return auc_resnet

def evaluate(model1, data_loader, loss_history):
    def threshold(ytrue, ypred):
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
        y = tpr - fpr
        youden_index = np.argmax(y)
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        return optimal_threshold, point, fpr, tpr


    model.eval()
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    people_id = []
    neg_pred_list = []
    labels = []

    with torch.no_grad():
        all_target = []
        all_prob = []
        for i, (id, data, target) in enumerate(data_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = F.log_softmax(model(data), dim=1)
            predicts = F.softmax(model(data), dim=1)
            neg_pred = predicts[:, 1]

            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            # print(output)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

            for x in target.detach().cpu():
                all_target.append(x)

            for y in predicts[:, 1].detach().cpu():
                all_prob.append(y)

            people_id.extend(id)
            neg_pred_list.extend(neg_pred.detach().cpu().numpy())
            labels.extend(target.detach().cpu().numpy())

    df = pd.DataFrame({'people_id': people_id, 'neg_preds': neg_pred_list, 'labels': labels})
    single_threshold, single_point, single_fpr, single_tpr = threshold(df['labels'], df['neg_preds'])

    auc_single = metrics.roc_auc_score(df['labels'], df['neg_preds'])
    df['single'] = (df['neg_preds'] >= single_threshold).astype(int)
    acc_single = (df['labels'] == df['single']).mean()
    df = df.groupby('people_id')[['labels', 'neg_preds']].mean()
    statistic_threshold, statistic_point, statistic_fpr, statistic_tpr = threshold(df['labels'], df['neg_preds'])
    df['outputs'] = (df['neg_preds'] >= statistic_threshold).astype(int)
    optimal_statistics_sensitivity = 1 - statistic_point[0]
    optimal_statistics_specificity = statistic_point[1]
    print("SE:", optimal_statistics_sensitivity)
    print("SP:", optimal_statistics_specificity)
    get_RoC(df)
    # auc_statis = metrics.roc_auc_score(df['labels'], df['neg_preds'])
    # acc_statistic = (df['labels'] == df['outputs']).mean()
    # get_RoC(df)


## Load Data
print('==> Preparing dataset')
Train_PATH = "/home/yangmy/MedTData/dataClean/cleaned/train"
Test_PATH = "/home/yangmy/MedTData/dataClean/cleaned/test"
train_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((64, 64)),
     torchvision.transforms.RandomGrayscale(p=0.5),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15, .15)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((64, 64)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = CCRFolder(Train_PATH, transform=train_transform)
test_dataset = CCRFolder(Test_PATH, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, drop_last=True)

if __name__ == '__main__':
    # model = Base_TransRseVit()
    model = Base_TransRegnet(num_class=2)
    # model = Base_Cait()
    # model = Base_TransSENet(num_class=2)
    # model = Base_ViT()
    # model = Base_TransDensenet()
    # model = Base_TransInception_v3()



    # model = se_resnet50(num_class=2)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(r"/home/yangmy/Code/VisualTrans/data_clean/save_model/ResVit_2_best.pth.tar")
    # checkpoint = torch.load(r"/home/yangmy/Code/VisualTrans/new_save_model/Densenet_best.pth.tar")
    # checkpoint = torch.load(r"/home/yangmy/Code/VisualTrans/new_save_model/Inception_best.pth.tar")
    # checkpoint = torch.load(r"/home/yangmy/Code/VisualTrans/new_save_model/Regnet_best.pth.tar")
    # checkpoint = torch.load(r"/home/yangmy/Code/VisualTrans/new_save_model/SENet_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    train_loss_history, test_loss_history = [], []
    best_signal_auc = 0
    best_signal_acc = 0
    best_person_auc = 0
    best_person_acc = 0
    evaluate(model, test_loader, test_loss_history)



























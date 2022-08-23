import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision
import PIL
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from variables import *
import numpy as np
from sklearn import metrics
from pre_models.MLP_mixer import Base_MLPMixer
from pre_models.ResViT import Base_Vitresnet
from pre_models.Cait import Base_Cait
from pre_models.densenet import Base_Densenet
from pre_models.inception import inception_v3
from pre_models.SENet import se_resnet50
from pre_models.regnet import regnet

import os
import argparse
import random
import pandas as pd
from dataloader import CCRFolder
import torch.utils.data as Data
from TransRestVit.utils import Bar, AverageMeter, accuracy
import itertools
from itertools import chain
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Hyper-Parameters")
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", "--learning-rate", default=1e-2, type=float)
parser.add_argument("--drop", "--dropout", default=0, type=float)
parser.add_argument('--schedule', type=int, default=[i * 50 for i in range(1, 6)])
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
parser.add_argument('--manualSeed', type=int, help='manual seed')

hy_args = parser.parse_args()
state = {k: v for k, v in hy_args._get_kwargs()}

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
use_cuda = torch.cuda.is_available()
if hy_args.manualSeed is None:
    hy_args.manualSeed = random.randint(1, 10000)
random.seed(hy_args.manualSeed)
torch.manual_seed(hy_args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(hy_args.manualSeed)

BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TRAIN_NUM = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_TEST_NUM = 100
N_EPOCHS = 300
Train_PATH = "/home/yangmy/MedTData/ccr_new/train"
Test_PATH = "/home/yangmy/MedTData/ccr_new/val"
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
class Df_Get():

    def __init__(self, args, root):
        self._root = root
        self._args = args
        self._state = {k: v for k, v in args._get_kwargs()}
        train_dataset = CCRFolder(Train_PATH, transform=train_transform)
        test_dataset = CCRFolder(Test_PATH, transform=test_transform)
        self._trainloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                            num_workers=4, drop_last=True)
        self._testloader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False,
                                           num_workers=4, drop_last=True)

        self._criterion = nn.CrossEntropyLoss()


    def model_loop(self):
        # upload the model with best auc
        ensemble_list_Resvit = ["ResVit_best.pth.tar"]
        ensemble_list_Cait = ["Cait_best_3.pth.tar"]
        ensemble_list_Densenet = ["Densenet_best.pth.tar"]
        ensemble_list_Inception = ["Inception_best_1.pth.tar"]
        ensemble_list_MLP = ["MLP_best.pth.tar"]
        ensemble_list_Regnet = ["Regnet_best_1.pth.tar"]
        ensemble_list_SENet = ["SENet_best_1.pth.tar"]

        df_list_vit = []
        for i_vit in ensemble_list_Resvit:
            model = Base_Vitresnet().cuda()
            model = torch.nn.DataParallel(model).cuda()
            pths_model_path = os.path.join(self._root, i_vit)
            state_dict = torch.load(pths_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model)
            df_list_vit.append((df, i_vit))

        df_list_cait = []
        for i_cait in ensemble_list_Cait:
            model = Base_Cait().cuda()
            model = torch.nn.DataParallel(model).cuda()
            pths_model_path = os.path.join(self._root, i_cait)
            state_dict = torch.load(pths_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model)
            df_list_cait.append((df, i_cait))

        df_list_inception = []
        for i_inception in ensemble_list_Inception:
            model = inception_v3().cuda()
            model = torch.nn.DataParallel(model).cuda()
            pths_model_path = os.path.join(self._root, i_inception)
            state_dict = torch.load(pths_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model)
            df_list_inception.append((df, i_inception))

        df_list_densenet = []
        for i_densenet in ensemble_list_Densenet:
            model = Base_Densenet().cuda()
            model = torch.nn.DataParallel(model).cuda()
            pths_model_path = os.path.join(self._root, i_densenet)
            state_dict = torch.load(pths_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model)
            df_list_densenet.append((df, i_densenet))

        df_list_mlp = []
        for i_mlp in ensemble_list_MLP:
            model = Base_MLPMixer().cuda()
            model = torch.nn.DataParallel(model).cuda()
            pths_model_path = os.path.join(self._root, i_mlp)
            state_dict = torch.load(pths_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model)
            df_list_mlp.append((df, i_mlp))

        df_list_regent = []
        for i_regent in ensemble_list_Regnet:
            model = regnet(num_class=2).cuda()
            model = torch.nn.DataParallel(model).cuda()
            pths_model_path = os.path.join(self._root, i_regent)
            state_dict = torch.load(pths_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model)
            df_list_regent.append((df, i_regent))

        df_list_senet = []
        for i_senet in ensemble_list_SENet:
            model = se_resnet50(num_class=2).cuda()
            model = torch.nn.DataParallel(model).cuda()
            pths_model_path = os.path.join(self._root, i_senet)
            state_dict = torch.load(pths_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model)
            df_list_senet.append((df, i_senet))

        df_list = {"resvit": df_list_vit, "cait": df_list_cait, "densenet": df_list_densenet,
                   "mlp": df_list_mlp, "inception": df_list_inception, "regnet": df_list_regent, "senet": df_list_senet}
        return df_list

    def test(self, model):
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        bar = Bar('Processing', max=len(self._testloader))
        people_id = []
        pred = []
        labels = []
        with torch.no_grad():
            for batch_idx, (id, data, target) in enumerate(self._testloader):
                inputs = data.float()
                targets = target.float()
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs = model(inputs)
                people_id.extend(id)
                pred.extend(F.softmax(outputs, -1)[:, 1].detach().cpu().numpy().tolist())
                labels.extend(targets.detach().cpu().numpy().tolist())
                loss = self._criterion(outputs, targets.long())
                prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0].item(), inputs.size(0))
                bar.suffix = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | acc1: {top1:.4f} | Loss: {loss_cifar:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(self._testloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1=top1.avg,
                    loss_cifar=losses.avg
                )
                bar.next()
            bar.finish()
            df = pd.DataFrame({'people_id': people_id, 'preds': pred, 'labels': labels})
            return df

class Df_Analysis():

    def __init__(self):
        self._df_list = Df_Get(hy_args, "./save_model").model_loop()
        self._df_list_vit = self._df_list["resvit"]
        self._df_list_cait = self._df_list["cait"]
        self._df_list_densenet = self._df_list["densenet"]
        self._df_list_mlp = self._df_list["mlp"]
        self._df_list_inception = self._df_list["inception"]
        self._df_list_regnet = self._df_list["regnet"]
        self._df_list_senet = self._df_list["senet"]


    def auc(self, df):
        def threshold(ytrue, ypred):
            fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
            y = tpr - fpr
            youden_index = np.argmax(y)
            optimal_threshold = thresholds[youden_index]
            point = [fpr[youden_index], tpr[youden_index]]
            roc_auc = metrics.auc(fpr, tpr)
            return optimal_threshold, point, fpr, tpr, roc_auc
        single_threshold, single_point, single_fpr, single_tpr, single = threshold(df['labels'], df['preds'])
        df['single'] = (df['preds'] >= single_threshold).astype(int)
        df = df.groupby('people_id')[['labels', 'preds']].mean()
        statistic_threshold, statistic_point, statistic_fpr, statistic_tpr, statis = threshold(df['labels'], df['preds'])
        df['outputs'] = (df['preds'] >= statistic_threshold).astype(int)
        acc_statistic = (df['labels'] == df['outputs']).mean()
        optimal_statistics_specificity = 1 - statistic_point[0]
        optimal_statistics_sensitivity = statistic_point[1]
        # print(statis, acc_statistic, optimal_statistics_specificity, optimal_statistics_sensitivity)
        return statis, acc_statistic, optimal_statistics_specificity, optimal_statistics_sensitivity

    def single_auc_get_sort(self):
        df_list_vit = []
        for i_vit in self._df_list_vit:
            auc, acc, specificity, sensitivity = self.auc(i_vit[0])
            print("vit: auc: {}, spe: {}, sen: {}".format(auc, specificity, sensitivity))
            df_list_vit.append((auc, i_vit))
        df_list_vit = sorted(df_list_vit, key=lambda x: x[0], reverse=True)
        df_list_cait = []
        for i_cait in self._df_list_cait:
            auc, acc, specificity, sensitivity = self.auc(i_cait[0])
            print("cait: auc: {}, spe: {}, sen: {}".format(auc, specificity, sensitivity))
            df_list_cait.append((auc, i_cait))
        df_list_cait = sorted(df_list_cait, key=lambda x: x[0], reverse=True)
        df_list_densenet = []
        for i_densenet in self._df_list_densenet:
            auc, acc, specificity, sensitivity = self.auc(i_densenet[0])
            print("densenet: auc: {}, spe: {}, sen: {}".format(auc, specificity, sensitivity))
            df_list_densenet.append((auc, i_densenet))
        df_list_densenet = sorted(df_list_densenet, key=lambda x: x[0], reverse=True)
        df_list_mlp = []
        for i_mlp in self._df_list_mlp:
            auc, acc, specificity, sensitivity = self.auc(i_mlp[0])
            print("mlp: auc: {}, spe: {}, sen: {}".format(auc, specificity, sensitivity))
            df_list_mlp.append((auc, i_mlp))
        df_list_mlp = sorted(df_list_mlp, key=lambda x: x[0], reverse=True)

        df_list_inception = []
        for i_inception in self._df_list_inception:
            auc, acc, specificity, sensitivity = self.auc(i_inception[0])
            print("inception: auc: {}, spe: {}, sen: {}".format(auc, specificity, sensitivity))
            df_list_inception.append((auc, i_inception))
        df_list_inception = sorted(df_list_inception, key=lambda x: x[0], reverse=True)

        df_list_regnet = []
        for i_regnet in self._df_list_regnet:
            auc, acc, specificity, sensitivity = self.auc(i_regnet[0])
            print("regent: auc: {}, spe: {}, sen: {}".format(auc, specificity, sensitivity))
            df_list_mlp.append((auc, i_regnet))
        df_list_regnet = sorted(df_list_regnet, key=lambda x: x[0], reverse=True)

        df_list_senet = []
        for i_senet in self._df_list_senet:
            auc, acc, specificity, sensitivity = self.auc(i_senet[0])
            print("senet: auc: {}, spe: {}, sen: {}".format(auc, specificity, sensitivity))
            df_list_senet.append((auc, i_senet))
        df_list_senet = sorted(df_list_senet, key=lambda x: x[0], reverse=True)

        return df_list_vit, df_list_cait, df_list_densenet, df_list_mlp, df_list_inception, df_list_regnet, df_list_senet

    def ensemble(self, used_value):
        df_list_vit, df_list_cait, df_list_densenet, df_list_mlp, df_list_inception, df_list_regnet, df_list_senet = self.single_auc_get_sort()
        df_list_vit_used = df_list_vit[0: used_value]
        df_list_cait_used = df_list_cait[0: used_value]
        df_list_densenet_used = df_list_densenet[0: used_value]
        df_list_mlp_used = df_list_mlp[0: used_value]
        df_list_inception_used = df_list_inception[0: used_value]
        df_list_regnet_used = df_list_regnet[0: used_value]
        df_list_senet_used = df_list_senet[0: used_value]

        df_list_total = []
        for i in df_list_vit_used:
            for j in df_list_cait_used:
                for k in df_list_densenet_used:
                    for m in df_list_mlp_used:
                        for l in df_list_inception_used:
                            for p in df_list_regnet_used:
                                for q in df_list_senet_used:
                                    df_list_total.append([i, j, k, m, l, p, q])
        df_list_chain = chain(df_list_vit, df_list_cait, df_list_densenet, df_list_mlp, df_list_inception, df_list_regnet, df_list_senet)
        df_list_total = itertools.combinations(df_list_chain, 2)

        max_auc = 0.0
        for i_iter in df_list_total:
            df_list_auc_cal = []
            df_list_df_cal = []
            df_list_path_cal = []
            for i_cal in i_iter:
                i_cal_df = i_cal[1][0]
                df_list_df_cal.append(i_cal_df)
                i_cal_auc = i_cal[0]
                df_list_auc_cal.append(i_cal_auc)
                i_cal_path = i_cal[1][1]
                df_list_path_cal.append(i_cal_path)
            print("The sequence of pths is {}".format(df_list_path_cal))
            print("The sequence of auc is {}".format(df_list_auc_cal))
            df_total = pd.concat(df_list_df_cal)
            auc, acc, specificity, sensitivity = self.auc(df_total)
            is_best = auc > max_auc
            if is_best:
                max_auc = max(max_auc, auc)
                print("The max auc is {}, the acc is {}, the specificity is {}, the sensitivity is {}".format(
                    max_auc, acc, specificity, sensitivity))
                print("The sequence of max auc is {}".format(df_list_path_cal))
        print("max auc", max_auc)

if __name__ == '__main__':
    Ensemble_Calculation = Df_Analysis()
    Ensemble_Calculation.ensemble(1)











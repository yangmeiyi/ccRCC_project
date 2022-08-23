# -*- coding: utf-8 -*-
"""
@author: meiyiyang
"""
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
import argparse
from torch import nn
import torch.nn.init as init
import torch.nn.parallel
from sklearn import metrics
from TransRestNet_model.utils import Bar, Logger, AverageMeter, accuracy
from dataloader import CCRFolder
import pandas as pd
import shutil
import os
import random
import numpy as np
from TransRestNet_model.py_identity import Identity
import torch


parser = argparse.ArgumentParser(description='PyTorch ccRCC Training')

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=100, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For small dataset uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  #

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class TransResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, dim=128, num_tokens=8, mlp_dim=256, heads=8, depth=12,
                 emb_dropout=0.0, dropout=0.0):
        super(TransResNet, self).__init__()
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 8x8 feature maps (64 in total)
        self.apply(_weights_init)

        # Tokenization

        self.token_wA = nn.Parameter(torch.empty(args.train_batch, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(args.train_batch, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=0.02)  # initialized based on the paper

        # self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)


        self.to_cls_token = Identity()

        self.nn1 = nn.Linear(dim, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        # self.af1 = GELU() # use additinal hidden layers only when training on large datasets
        # self.do1 = nn.Dropout(dropout)
        # self.nn2 = nn.Linear(mlp_dim, num_classes)
        # torch.nn.init.xavier_uniform_(self.nn2.weight)
        # torch.nn.init.normal_(self.nn2.bias)
        # self.do2 = nn.Dropout(dropout)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, img, mask=None):
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = rearrange(x,
                      'b c h w -> b (h w) c')  # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP  把图片一维展开
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose'
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)
        # x = self.af1(x)
        # x = self.do1(x)
        # x = self.nn2(x)
        # x = self.do2(x)
        return x


def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    for i, (id, data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        acc = accuracy(output.data, target.data, topk=(1,))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) + ' Acc:' + '{:6.2f}'.format(acc[0].item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
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
            # print(output)
            predicts = F.softmax(model(data), dim=1)
            neg_pred = predicts[:, 1]

            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

            for x in target.detach().cpu():
                all_target.append(x)

            for y in predicts[:, 1].detach().cpu():
                all_prob.append(y)

            people_id.extend(id)
            neg_pred_list.extend(neg_pred.detach().cpu().numpy().tolist())
            labels.extend(target.detach().cpu().numpy().tolist())

    df = pd.DataFrame({'people_id': people_id, 'neg_preds': neg_pred_list, 'labels': labels})
    newindex = people_id
    df.index = newindex
    acc_single, acc_statistic, auc_single, auc_statis, single_threshold, statistic_threshold, \
    single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point = Auc(df)
    single_sensitivity = 1 - single_fpr
    single_specificity = single_tpr
    optimal_single_sensitivity = 1 - single_point[0]
    optimal_single_specificity = single_point[1]
    statistics_sensitivity = 1 - statistic_fpr
    statistics_specificity = statistic_tpr
    optimal_statistics_sensitivity = statistic_point[0]
    optimal_statistics_specificity = statistic_point[1]

    acc = 100.0 * correct_samples / total_samples
    auc = metrics.roc_auc_score(all_target, all_prob)

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)

    print('Average test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)' + ' Auc: {}\n'.format(auc))

    return avg_loss, acc_single, acc_statistic, auc_single, auc_statis

def Auc(df):
    def threshold(ytrue, ypred):
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
        y = tpr - fpr
        youden_index = np.argmax(y)
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        print(optimal_threshold)
        return optimal_threshold, point, fpr, tpr

    single_threshold, single_point, single_fpr, single_tpr = threshold(df['labels'], df['neg_preds'])

    auc_single = metrics.roc_auc_score(df['labels'], df['neg_preds'])
    df['single'] = (df['neg_preds'] >= single_threshold).astype(int)
    acc_single = (df['labels'] == df['single']).mean()
    df = df.groupby('people_id')[['labels', 'neg_preds']].mean()
    statistic_threshold, statistic_point, statistic_fpr, statistic_tpr = threshold(df['labels'], df['neg_preds'])
    df['outputs'] = (df['neg_preds'] >= statistic_threshold).astype(int)
    auc_statis = metrics.roc_auc_score(df['labels'], df['neg_preds'])
    acc_statistic = (df['labels'] == df['outputs']).mean()
    return acc_single, acc_statistic, auc_single, auc_statis, single_threshold, statistic_threshold, \
           single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point







def save_checkpoint(state, is_best, path, filename):
    checkpoint = path
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'TransResNet_best.pth.tar'))



if __name__ == '__main__':
    print('==> Preparing dataset')
    path = "/home/yangmy/Code/data_clean/save_model/"
    filename = "TransResNet"

    Train_PATH = "/home/yangmy/MedTData/dataClean/cleaned/train"
    Test_PATH = "/home/yangmy/MedTData/dataClean/cleaned/test"
    trainTransform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((64, 64)),
         torchvision.transforms.RandomGrayscale(p=0.5),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
         torchvision.transforms.RandomAffine(8, translate=(.15, .15)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testTransform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((64, 64)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_dataset = CCRFolder(Train_PATH, transform=trainTransform)
    test_dataset = CCRFolder(Test_PATH, transform=testTransform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, drop_last=True)

    model = TransResNet(BasicBlock, [3, 4, 6])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=.9, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50,100,150], gamma=0.1)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    train_loss_history, test_loss_history = [], []
    best_signal_auc = 0
    best_signal_acc = 0
    best_person_auc = 0
    best_person_acc = 0

    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        vg_loss, acc_single, acc_statistic, auc_single, auc_statis = evaluate(model, test_loader, test_loss_history)
        best_signal_auc = max(best_signal_auc, auc_single)
        is_best = auc_statis > best_person_auc
        best_person_auc = max(best_person_auc, auc_statis)
        best_signal_acc = max(best_signal_acc, acc_single)
        best_person_acc = max(best_person_acc, acc_statistic)

        # save model
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'auc': best_person_auc,
            'acc': best_person_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, path, filename)
        print("best_signal_auc: {}".format(best_signal_auc) + " best_person_auc: {}".format(best_person_auc) + " best_signal_acc: {}".format(best_signal_acc) + " best_person_acc: {}\n".format(best_person_acc))
        lr_scheduler.step()








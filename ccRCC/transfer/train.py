import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import shutil
import PIL
from dataloader import CCRFolder
import pandas as pd
import random
from sklearn import metrics
import torchvision
from TransRestNet_model.utils import *
from cnn_finetune import make_model



os.environ['CUDA_VISIBLE_DEVICES'] = "1"

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

use_cuda = torch.cuda.is_available()

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
    optimal_statistics_sensitivity =1- statistic_point[0]
    optimal_statistics_specificity = statistic_point[1]

    acc = 100.0 * correct_samples / total_samples
    auc = metrics.roc_auc_score(all_target, all_prob)

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)

    print("SE", optimal_statistics_sensitivity)
    print("SP", optimal_statistics_specificity)

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
        # print(optimal_threshold)
        return optimal_threshold, point, fpr, tpr

    single_threshold, single_point, single_fpr, single_tpr = threshold(df['labels'], df['neg_preds'])

    auc_single = metrics.roc_auc_score(df['labels'], df['neg_preds'])
    df['single'] = (df['neg_preds'] >= single_threshold).astype(int)
    acc_single = (df['labels'] == df['single']).mean()
    df = df.groupby('people_id')[['labels', 'neg_preds']].mean()
    # df = df.groupby('people_id')[['labels', 'neg_preds']].mean()
    statistic_threshold, statistic_point, statistic_fpr, statistic_tpr = threshold(df['labels'], df['neg_preds'])
    df['outputs'] = (df['neg_preds'] >= statistic_threshold).astype(int)
    auc_statis = metrics.roc_auc_score(df['labels'], df['neg_preds'])
    # print(df['labels'][:20], df['neg_preds'][:20])
    acc_statistic = (df['labels'] == df['outputs']).mean()
    return acc_single, acc_statistic, auc_single, auc_statis, single_threshold, statistic_threshold, \
           single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point

def save_checkpoint(state, is_best, checkpoint= "./save_model/", filename='Densenet.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'Densenet_best.pth.tar'))


gpu_num = 2
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TRAIN_NUM = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_TEST_NUM = 100
N_EPOCHS = 100

print('==> Preparing dataset')
Train_PATH = "/home/yangmy/MedTData/dataClean/cleaned/train/"
Test_PATH = "/home/yangmy/MedTData/dataClean/cleaned/test/"
train_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((64, 64)),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15, .15)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((64, 64)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# train_dataset = torchvision.datasets.ImageFolder(Train_PATH, transform=transform)
# test_dataset = torchvision.datasets.ImageFolder(Test_PATH, transform=transform)
train_dataset = CCRFolder(Train_PATH, transform=train_transform)
test_dataset = CCRFolder(Test_PATH, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, drop_last=True)




if __name__ == '__main__':
    # model = make_model('resnet18', num_classes=2, pretrained=True)
    model = make_model('densenet121', num_classes=2, pretrained=True)
    # model = make_model('senet154', num_classes=2, pretrained=False)
    # model = make_model('alexnet', num_classes=2, pretrained=False, input_size=(64, 64))
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=.9, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[25,50,75], gamma=0.1)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    train_loss_history, test_loss_history = [], []
    best_signal_auc = 0
    best_signal_acc = 0
    best_person_auc = 0
    best_person_acc = 0

    for epoch in range(1, N_EPOCHS + 1):
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
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'auc': best_person_auc,
        #     'acc': best_person_acc,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best)
        print("best_signal_auc: {}".format(best_signal_auc) + " best_person_auc: {}".format(best_person_auc) + " best_signal_acc: {}".format(best_signal_acc) + " best_person_acc: {}\n".format(best_person_acc))
        lr_scheduler.step()








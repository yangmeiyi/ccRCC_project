# Import Libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt



def get_RoC(resnet_fpr, resnet_tpr, DenseNet_fpr, DenseNet_tpr, Inception_fpr, Inception_tpr, Regnet_fpr, Regnet_tpr, SENet_fpr, SENet_tpr):
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

    plt.tick_params(axis='x', width=3, direction='out', labelsize=18)
    plt.tick_params(axis='y', width=3, direction='out', labelsize=18)
    plt.plot(resnet_fpr, resnet_tpr, linewidth=3, color='#FF464F', label='TransResNet(area = 0.920)')
    plt.plot(DenseNet_fpr, DenseNet_tpr, linewidth=3, color='#569DFF', label='TransDenseNet(area = 0.890)')
    plt.plot(Inception_fpr, Inception_tpr, linewidth=3, color='#FF7A0E', label='TransInception(area = 0.881)')
    plt.plot(Regnet_fpr, Regnet_tpr, linewidth=3, color='#75FF6C', label='TransRegnet(area = 0.877)')
    plt.plot(SENet_fpr, SENet_tpr, linewidth=3, color='#FF6BFF', label='TransSENet(area = 0.895)')
    # plt.fill_between(fpr, tpr, interpolate=True, color='#FF9492', alpha=0.5)
    plt.legend(loc='lower right', fontsize=16)


    plt.plot(x, y, linewidth=4, color='gray', linestyle='--')
    plt.grid()
    plt.savefig("./roc.pdf", dpi=1000)
    plt.show()

if __name__ == '__main__':
    resnet_fpr = np.load('resnet_fpr.npy')
    resnet_tpr = np.load('resnet_tpr.npy')
    DenseNet_fpr = np.load('DenseNet_fpr.npy')
    DenseNet_tpr = np.load('DenseNet_tpr.npy')
    Inception_fpr = np.load('Inception_fpr.npy')
    Inception_tpr = np.load('Inception_tpr.npy')
    Regnet_fpr = np.load('Regnet_fpr.npy')
    Regnet_tpr = np.load('Regnet_tpr.npy')
    SENet_fpr = np.load('SENet_fpr.npy')
    SENet_tpr = np.load('SENet_tpr.npy')
    get_RoC(resnet_fpr, resnet_tpr, DenseNet_fpr, DenseNet_tpr, Inception_fpr, Inception_tpr, Regnet_fpr, Regnet_tpr, SENet_fpr, SENet_tpr)


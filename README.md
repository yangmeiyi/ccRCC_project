# TransResNet

by Meiyi Yang, Xiaopeng He, Lifeng Xu, Minghui Liu6, Jiali Deng, Xuan Cheng, Yi Wei4, Qian Li, Shang Wan, Feng Zhang, Xiaomin Wang, Lei Wu, Bin Song，Ming Liu.

## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".

TransResNet was proposed to predict the low and high nuclear grade ccRCC, which combines the advantages of Transformer and ResNet. The proposed TransResNet model obtained an average accuracy of 87.1%, a sensitivity of 91.3%, a specificity of 85.3%, and an area under the curve (AUC) of 90.3% on the collected ccRCC data set containing 759 patients.

## 环境依赖


## 目录结构描述
├──Readme.md                // help  <br>
├──ccRCC              <br>     
&emsp ├──external             // external validation (from public dataset TCGA-KIRC)  <br>
&emsp ├── fpr_tpr_data             // tpr and fpr data, which can be used to draw ROC curves  <br>
> > ├── roc_curve            //  ROC curves  <br>
> > ├── image <br>
> > ├── pre_models // models for integration and comparison  <br>
> > ├──transger // transfer learning  <br>
> > ├── TransResNet_model // TransResNet training  <br>
> > ├── data_enhancement // TransResNet training  <br>
> > ├── ensemble // model based on ensemble learning  <br>
> > ├── heat_map // heat map  <br>
> > ├── hot_view // CAM <br>



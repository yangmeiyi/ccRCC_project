# TransResNet

by Meiyi Yang, Xiaopeng He, Lifeng Xu, Minghui Liu6, Jiali Deng, Xuan Cheng, Yi Wei4, Qian Li, Shang Wan, Feng Zhang, Xiaomin Wang, Lei Wu, Bin Song，Ming Liu.

## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".

TransResNet was proposed to predict the low and high nuclear grade ccRCC, which combines the advantages of Transformer and ResNet. The proposed TransResNet model obtained an average accuracy of 87.1%, a sensitivity of 91.3%, a specificity of 85.3%, and an area under the curve (AUC) of 90.3% on the collected ccRCC data set containing 759 patients.

## 环境依赖


## 目录结构描述
> ├──Readme.md        &emsp;&emsp; &emsp; &emsp;&emsp; &emsp;        // help  <br>
> ├──ccRCC              <br>  
> > ├──external       &emsp;&emsp; &emsp;         // external validation (from public dataset TCGA-KIRC)  <br>
> > ├── fpr_tpr_data     &emsp;&emsp; &emsp;           // tpr and fpr data, which can be used to draw ROC curves  <br>
> > ├── roc_curve      &emsp;&emsp; &emsp;         //  ROC curves  <br>
> > ├── image <br>
> > ├── pre_models  &emsp;&emsp; &emsp;         // models for integration and comparison  <br>
> > ├──transger  &emsp;&emsp; &emsp;              // transfer learning  <br>
> > ├── TransResNet_model  &emsp;&emsp; &emsp;   // TransResNet training  <br>
> > ├── data_enhancement  &emsp;&emsp; &emsp;   // TransResNet training  <br>
> > ├── ensemble  &emsp;&emsp; &emsp;   // model based on ensemble learning  <br>
> > ├── heat_map  &emsp;&emsp; &emsp;   // heat map  <br>
> > ├── hot_view  &emsp;&emsp; &emsp;   // CAM <br>



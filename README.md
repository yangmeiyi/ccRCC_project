# TransResNet

by Meiyi Yang, Xiaopeng He, Lifeng Xu, Minghui Liu6, Jiali Deng, Xuan Cheng, Yi Wei4, Qian Li, Shang Wan, Feng Zhang, Xiaomin Wang, Lei Wu, Bin Song，Ming Liu.

## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".
![image](https://github.com/yangmeiyi/ccRCC_project/blob/main/network.jpg)

TransResNet was proposed to predict the low and high nuclear grade ccRCC, which combines the advantages of Transformer and ResNet. The proposed TransResNet model obtained an average accuracy of 87.1%, a sensitivity of 91.3%, a specificity of 85.3%, and an area under the curve (AUC) of 90.3% on the collected ccRCC data set containing 759 patients.




## Content
> ├──Readme.md               // help  <br>
> ├──ccRCC              <br>  
> > ├──external             // external validation (from public dataset TCGA-KIRC)  <br>
> > ├── fpr_tpr_data        // tpr and fpr data, which can be used to draw ROC curves  <br>
> > ├── roc_curve           //  ROC curves  <br>
> > ├── image <br>
> > ├── pre_models          // models for integration and comparison  <br>
> > ├──transfer             // transfer learning  <br>
> > ├── TransResNet_model                      // TransResNet training  <br>
> > ├── ensemble     // model based on ensemble learning  <br>
> > ├── heat_map    // heat map  <br>
> > ├── hot_view    // CAM <br>


## Code 

### Requirements
* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.6.8
* PyTorch >= 1.0.1
* torchvision
* einops
* cuda

### Parameters
![image](https://github.com/yangmeiyi/ccRCC_project/blob/main/parameters.png)
<div align=center><img src="https://github.com/yangmeiyi/ccRCC_project/blob/main/parameters.png" width="200" height="100" /></div>

### Usage

python3 ./TransResNet_model/train.py

### Getting Started
* For TransResNet, please see TransResNet_model/train for detailed instructions.
* For Transfer Learnig, please see Transfer.
* For Ensembel learning, please see re_models and ensemble.

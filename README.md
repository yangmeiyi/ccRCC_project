# TransResNet

by Meiyi Yang, Xiaopeng He, Lifeng Xu, Minghui Liu6, Jiali Deng, Xuan Cheng, Yi Wei4, Qian Li, Shang Wan, Feng Zhang, Xiaomin Wang, Lei Wu, Bin Song，Ming Liu.

## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".

TransResNet was proposed to predict the low and high nuclear grade ccRCC, which combines the advantages of Transformer and ResNet. The proposed TransResNet model obtained an average accuracy of 87.1%, a sensitivity of 91.3%, a specificity of 85.3%, and an area under the curve (AUC) of 90.3% on the collected ccRCC data set containing 759 patients.

## 环境依赖


## 目录结构描述
> ├── Readme.md                   // help
> > ├── Readme.md                   // help
> > > ├── Readme.md                   // help
> > > 
├── app                         // 应用
├── config                      // 配置
│   ├── default.json
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境
├── data
├── doc                         // 文档
├── environment
├── gulpfile.js
├── locales
├── logger-service.js           // 启动日志配置
├── node_modules
├── package.json
├── app-service.js              // 启动应用配置
├── static                      // web静态资源加载
│   └── initjson
│       └── config.js         // 提供给前端的配置
├── test
├── test-service.js
└── tools

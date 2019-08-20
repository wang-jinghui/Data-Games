# 比赛地址
https://dianshi.baidu.com/competition/30/rank

初赛：95，复赛：10

团队成员：

 - 机器要学习：me
 - Stone : https://github.com/stormstone
 - 买书淘书
 - sky未完_待续
    
---------------------------------------------
### 系统环境

os : ubuntu 18.04  server

##### python版本及第三方的库

###### GPU环境：

Python  3.6.8

pandas  0.24.2

scikit-image  0.15.0

scikit-learn  0.20.3

tensorflow-gpu  1.12.0

tqdm 4.31.1

xgboost  0.90

numpy  1.16.3

lightgbm 2.2.3

------------

###### CPU环境

python  3.7.2

pandas  0.24.2

scikit-image  0.14.2

scikit-learn  0.20.2

tensorflow  1.13.1

tqdm 4.31.1

xgboost  0.80

numpy  1.14.5

lightgbm 2.2.1

### <font color='red'>以上两个环境可任意切换</font>

### 目录结构

当前目录下有四个文件夹，分别是：

code   data   features   result
目录结构如下所示：

    ├── code                                                 // 脚本文件
    ├── data                                                 // 原始数据
    │   ├── train_image                                      // 训练图像数据
    │   │   ├── 001
    │   │   ├── 002
    │   │   ├── 003
    │   │   ├── 004
    │   │   ├── 005
    │   │   ├── 006
    │   │   ├── 007
    │   │   ├── 008
    │   │   └── 009 
    │   ├── train_visit
    │   │   ├── 0
    │   │   ├── 1
    │   │   ├── 2
    │   │   ├── 3
    │   │   ├── 4
    │   │   ├── 5
    │   │   ├── 6
    │   │   ├── 7
    │   │   ├── 8
    │   │   └── 9 
    │   ├── test_image   
    │   ├── test_visit
    │   │   ├── 0
    │   │   └── 1                   
    ├── features                      
    ├── result                       

code : 所有的脚本(均有同名的Jupyter notebook格式文件)

- preprocess_data.py
- CNN.py
- MLP_XGB_LGB.py
- Stacking.py

data : 原始的数据, 下面有四个子文

- train_image ：训练图像
- train_visit      : 训练日志数据
- test_image    : 测试图像
- test_visit        : 测试集日志数据

features : 中间结果和特征文件

result : 结果文件

---

# 脚本执行顺序

1. preprocess_data.py ，数据的预处理和特征提取，结果放在'../features/'目录下。
2. CNN.py，图像数据的训练，生成结果文件在'../features/'目录下。
3. MLP_XGB_LGB.py，模型的训练，生成结果文件在'../features/'目录下。
4. Stacking.py，模型结果的集成，读取'../features/'目录下的中间结果，生成最终提交文件。

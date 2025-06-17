# Pix2Pix_Facades
Pix2Pix图像生成网络训练Facades数据集

# 文件结构  格式乱的话改成Code模式观看
└─RUN_1
    │  INFO.txt
    │
    ├─D/
    │      dict_epoch_120.pth
    │      dict_epoch_168.pth
    │
    ├─fake_imgs/
    │      epoch_1.jpg
    │      epoch_10.jpg
    │      epoch_100.jpg
    │
    └─G/
            dict_epoch_120.pth
            dict_epoch_168.pth

# 数据集请见
https://aistudio.baidu.com/datasetdetail/230639

# D/
存放鉴别器权重

# G/
存放生成器权重

# fake_imgs
epoch生成的虚假图

# Functions.py
封装一些功能函数，只在datasets_Facades.py中使用了Functions.py中封装的函数

# INFO.txt
保存训练过程中的参数信息

# Main.py
主训练文件

# Pix2Pix.py
模型结构信息

# ReadInfo.py
用于可视化INFO.txt内容，例如学习率曲线，损失曲线

# datasets_Facades.py
Facades数据集加载器








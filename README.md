Provide Chinese annotations for the main functions of the Faceswap_Deepfake project by Oldpan


原来的项目地址：https://github.com/Oldpan/Faceswap-Deepfake-Pytorch
增加Oldpan/Faceswap-Deepfake-Pytorch的一些主要功能的代码中文注释，方便初学理解这些功能的作用之类的

作者有提供数据，挺不错的
如果你有跟土堆学习过pytorch：https://www.bilibili.com/video/BV1hE411t7RN/
应该看得懂我的注释。

面向deepfake,深伪检测的初学者使用

将这个链接下载https://1drv.ms/u/c/52986a6839d1d98b/EdzJGVfUzVxDhpLctJgEqTABr137doGaUk2EDAKcgvXb-g?e=kcKeNT
其中数据集放在data,预训练好的模型放在checkpoint（注：作者提供了预训练好的模型，
如果你不想要预训练，也可以直接自己训练，但是需要在train.py的101行把“start_epoch"字段删掉，改为0。这样他才能从头开始训练，而不是检查预训练的start_epoch）


# Guidance of CLIP-ReID for Test on MSMT17

Tips：该代码仓库继承于Trans-reid，如果后续深度使用觉得比较麻烦，可以参考Trans-reid的代码，它的代码比较简洁。

## 数据集准备

代码解压后，除了meta文本文件以及图像数据本体外，还有一个python转换脚本。该转换脚本用于将从[msmt17_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/166508)上下载的MSMT17转换为目前代码能运行的格式。准备如下：

- 从该链接下载数据集后应该可以得到一个MSMT17.zip文件，解压该文件:

  ````
  unzip MSMT17.zip -d MSMT17
  ````

- 运行msmt17_format_trans.py即可得到符合运行标准的数据集。该脚本作用：

  - 为数据集生成指定meta文件；
  - 按照一定比率将train数据集分割成train和val；
  - 重命名图像文件夹。

- 最终的数据集目录结构为（_MACOSX无用）：

  ![image-20241202194230767](command/image-20241202194230767.png)

## 环境准备

解压压缩包后，根据README.MD安装相应包，配置好相关环境。此外还需要从[MSMT17_clipreid_ViT-B-16_60.pth - Google 云端硬盘](https://drive.google.com/file/d/1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO/view)下载权重文件。其他model/log可见README.md的表格。

## 代码架构简析

```
CLIP-ReID
├── config
├── configs
│   ├── person
│   ├── VehicleID
│   └── veri
├── datasets
├── loss
├── model
│   ├── clip
├── processor
├── solver
└── utils
```

代码仓库中的主要文件夹如上所示，configs中为配置yaml文件，其他均为提前编写好的对应模块包文件夹(被引用)。

本次测试主要使用的是tools/msmt17_format_trans.py、test_clipreid.py、nohup.sh以及configs/person/vit_clipreid.yml四个文件。

## 参数配置

路径修改：（见README.md的Training部分）在vit_clipreid.yml中修改为自己的数据集路径。

其他配置：类似于图片输入大小、测试的batch size等均在vit_clipreid.yml中配置即可。

## 开始训练

切换目录

```shell
cd CLIP-ReID
```

运行bash文件

```shell
nohup bash nohup.sh > output.log 2>&1 &
```

在nohup.sh中可以修改gpu-idx以及一些运行时的相关参数（建议只修改CUDA_VISIBLE_DEVICES）

```bash
CUDA_VISIBLE_DEVICES=2,3 python test_clipreid.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/data/jhb_data/checkpoints/MSMT17_clipreid_ViT-B-16_60.pth'#这里换成权重文件路径
```

## 测试汇报

输入图片大小：(256, 128)

batch size：64

Model size: 487.28 MB

每个batch推理（生成特征向量）所耗的平均时间：0.005436420440673828 second.

验证精度结果：
2024-12-02 03:49:34,492 transreid.test INFO: mAP: 73.3%
2024-12-02 03:49:34,492 transreid.test INFO: CMC curve, Rank-1  :88.7%
2024-12-02 03:49:34,492 transreid.test INFO: CMC curve, Rank-5  :94.4%
2024-12-02 03:49:34,492 transreid.test INFO: CMC curve, Rank-10 :95.7%

## Contact me

wechat: FisherVEM


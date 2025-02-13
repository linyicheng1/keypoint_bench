# 图像点特征性能指标对比

### 支持的点特征
- ORB
- SIFT
- Harris 
- SuperPoint
- ALIKE
- XFeat
- D2-Net
- DISK
- R2D2
- SFD2

### 支持的匹配方法
- 暴力匹配
- lightglue
- 光流

### 评价指标
- AUC  基于 MegaDepth 数据集
- MHA  基于 Hpatch 数据集
- 重复率 基于 Hpatch 数据集
- 基础矩阵 基于TartanAir数据集
- VO里程计 基于KITTI数据集

### 一些结果

#### 参数

> 值得注意的是，许多论文为了指标更高，往往提取更多的特征，并且非极大值抑制范围很小，这样在实际使用的时候会造成特征扎堆的现象。因此，我们使用的参数都是特征点数量较少，并且非极大值抑制的窗口比较大，这更符合实际情况。**尽管这样往往会导致指标较低。**

##### 特征提取参数
- nms_dist: 6
- min_score: 0.0
- top_k: 500
- threshold: 0
- border_dist: 8

##### 暴力匹配参数

- metric: euclidean
- max_distance: 5
- cross_check: True

##### 光流匹配参数

- distance: 10
- win_size: 21
- levels: 3
- interation: 40

#### 结果

##### 重复率 

| 特征点 | Harris | ALIKE | SuperPoint | XFeat | D2-Net | DISK | R2D2 | SFD2 | LGood (Ours) | EdgePoint (Ours) |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 重复率 |  0.139 | 0.315 | 0.375 | 0.125 |  0.142 | 0.290 | 0.378 | 0.428 | 0.345 | 0.345 | 0.353|
| 平均误差 | 1.851 | 1.231 | 1.320 | 1.389 | 2.011 | 1.473 | 1.417 | 1.28 | 1.294 | 1.274|

##### MHA

##### AUC

##### 


## RUN IN Docker 

1. 构造docker镜像

```shell
mkdir keypoint_bench && cd keypoint_bench
wget https://github.com/linyicheng1/keypoint_bench/blob/main/Dockerfile
docker build -t keypoint_bench:v1 .
```

需要等待10-20分钟，下载依赖并且构建镜像

2. 运行docker容器

```shell
docker run -it --gpus all -u root -p 2223:22 -v /数据集下载位置/:/home/data/ keypoint_bench:v1
```

3. 运行python程序

在运行程序前需要修改配置文件中包含的数据集路径，按自己的实际情况修改`/home/code/keypoint_bench/config/config.yaml`文件


运行程序
```shell
cd /home/code/keypoint_bench
python3 main.py --config_file config/config.yaml test 
```

默认得到ALIKE的重复率计算结果

repeatability 0.3157695  rep_mean_err 1.2313193






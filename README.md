# Travel around the world photographically

## Project Introduction

本项目提出于疫情中出游不便的时代背景下，主要聚焦于如何使戴口罩的照片变得生动多彩。项目算法由以下部分组成：

1. 利用Matting算法实现人物前景和背景的自动分割；
2. 借助人脸识别和closed-form matting进行口罩的自动识别分割；
3. 根据自动分割出的口罩区域，利用基于梯度选择的改进版泊松扩散，添加用户自选花纹图样；
4. 使用泊松扩散实现人物前景与目标背景的自然融合。

我们的实验结果表明，我们的方法可以很好地适用于不同环境图片和人物图片的融合，可以自然地保留口罩的颜色、褶皱等关键特征，最大程度减少了口罩失真；同时在第四部分中通过对原背景图的高斯模糊和翻转，缓和了背景色调不均一对融合自然度的影响。

## How to run

环境搭建：

```shell
python==3.8
pip install numpy
pip install scipy
pip install matplotlib
pip install torch
pip install torchvision
pip install opencv-python
pip install face-recognition
```

- PS：在Windows系统上，直接`pip install face-recognition`可能无法安装成功，可以根据[该说明](https://www.geeksforgeeks.org/how-to-install-face-recognition-in-python-on-windows/)进行安装

运行生成：

```shell
# example command 1
python main.py -s figs/example/source1.jpg -t figs/example/target1.jpg -p figs/example/pattern1.jpg -r 0.5 -a -g 755
# example command 2
python main.py -a -s figs/test/source_1.jpg -t figs/test/target.jpg -r 0.5 -l 1.1

# 参数解释
# -s: str,      含前景人像的图片路径
# -t: str,      作为背景的图片路径
# -p: str,      作为口罩图案的图片路径，如果不指定，则不改变口罩图案
# -r: float,    为提高运算速度，指定图片缩放的倍数
# -a: flag,     如果指定，启用MODNet自动分割；否则，根据API手动划分区域
# -g: int,      指定高斯核的大小，需要是奇数
# -l: float,    调整图片亮度
# -d: flag,     如果指定，对人物和背景图做直接拼接而非poisson blending
```

## Example

### Running Example

<video src="./pic/example_running.mp4"></video>

### Different Environment

<table><tr>
<td style="width:50%">
    <center>
        <img src=./pic/bright.png border=0>
        <p>fig.1 明亮环境</p>
    </center>
</td>
<td style="width:50%">
    <center>
        <img src=./pic/dark.png border=0>
        <p>fig.2 昏暗环境</p>
    </center>
</td>
</tr></table>

<table><tr>
<td style="width:50%">
    <center>
        <img src=./pic/day.png border=0>
        <p>fig.3 白天</p>
    </center>
</td>
<td style="width:50%">
    <center>
        <img src=./pic/night.jpg border=0>
        <p>fig.4 夜晚</p>
    </center>
</td>
</tr></table>

<table><tr>
<td style="width:50%">
    <center>
        <img src=./pic/indoor.png border=0>
        <p>fig.5 室内</p>
    </center>
</td>
<td style="width:50%">
    <center>
        <img src=./pic/outdoor.png border=0>
        <p>fig.6 室外</p>
    </center>
</td>
</tr></table>

<table><tr>
<td style="width:50%">
    <center>
        <img src=./pic/sin.png border=0>
        <p>fig.7 单人</p>
    </center>
</td>
<td style="width:50%">
    <center>
        <img src=./pic/multi.png border=0>
        <p>fig.8 多人</p>
    </center>
</td>
</tr></table>


## Acknowledgements

我们的模型参考了以下代码项目仓库：
- [MODNet](https://github.com/ZHKKKe/MODNet)
- [closed-form-matting](https://github.com/MarcoForte/closed-form-matting)
- [poisson-image-editing](https://github.com/PPPW/poisson-image-editing)
- [face_recognition](https://github.com/ageitgey/face_recognition)

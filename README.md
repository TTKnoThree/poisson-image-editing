# Poisson Image Editing to Mask Travel

当前框架：

1. 用户输入原照片与目标背景，MODNet自动分割原照片人物（已完成）
2. 用户指定摆放位置，poisson进行人物背景融合（不够理想，进一步实现原论文中改进向量场）
3. facial recognition + trimap matting自动分割口罩(已完成)
4. 用户上传替换图案，poisson进行口罩替换（对偶任务，创新）
5. 用户得到输出图片

## How to run

```shell
python==3.8
pip install numpy
pip install scipy
pip install matplotlib
pip install torch
pip install torchvision
pip install opencv-python
pip install face_recognition
```

```shell
python main.py -s figs/example/source1.jpg -t figs/example/target1.jpg -p figs/example/pattern1.jpg -r 0.5 -a
python main.py -a -s figs/test/source_1.jpg -t figs/test/target.jpg -r 0.5

# 参数解释
# -s 含前景人像的图片路径
# -t 作为背景的图片路径
# -p 作为口罩图案的图片路径，如果不指定，则不改变口罩图案
# -r 为提高运算速度，图片缩放的倍数
# -a 启用MODNet自动分割
```

## Mask Separation

- Requirement: face-recognition
- PS：在Windows系统上，直接pip install face-recognition可能无法安装成功，可以根据以下链接安装，要求python版本为3.7或3.8
- [在Windows上安装face-recognition包的方法](https://www.geeksforgeeks.org/how-to-install-face-recognition-in-python-on-windows/)
- mask_separation/detect_mask.py用法：

  ```shell
  python mask_separation/detect_mask.py -i <path_of_input_image>
  python mask_separation/detect_mask.py -i <path_of_input_image> --resize --resize_t 0.4
  ```

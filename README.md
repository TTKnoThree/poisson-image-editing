# Poisson Image Editing to Mask Travel

当前框架：

1. 用户输入原照片与目标背景，MODNet自动分割原照片人物（已完成）
2. 用户指定摆放位置，poisson进行人物背景融合（不够理想，进一步实现原论文中改进向量场）
3. facial recognition + trimap matting自动分割口罩(已完成)
4. 用户上传替换图案，poisson进行口罩替换（对偶任务，创新）
5. 用户得到输出图片

## How to run

```
python main.py -s figs/test/source.png -t figs/test/target.jpg -m figs/test/matte.png
python main.py -a 1 -s figs/test/1_source.png -t figs/test/target.jpg

# 参数解释
# -a 启用MODNet自动分割，后面的参数随便填
# -s 含前景人像的图片路径
# -t 作为背景的图片路径
# -m 如果已有matting（黑白底），则输入matte的路径
```

## Mask Separation

- Requirement: face_alignment
- PS：在Windows系统上，直接pip install face_alignment可能无法安装成功，可以根据以下链接安装，要求python版本为3.7或3.8
- [在Windows上安装face-recognition包的方法](https://www.geeksforgeeks.org/how-to-install-face-recognition-in-python-on-windows/)
- mask_separation/detect_mask.py用法：

  ```shell
  python mask_separation\detect_mask.py -i mask_separation\detect_mask.py
  python mask_separation\detect_mask.py -i mask_separation\detect_mask.py --resize --resize_t 0.4
  ```

import numpy
from image_augmentation import random_transform
from image_augmentation import random_warp

random_transform_args = {
    'rotation_range': 10, # 随机旋转角度范围
    'zoom_range': 0.05, # 随机缩放的比例
    'shift_range': 0.05,# 随机平移的比例
    'random_flip': 0.4,# 随机水平翻转的概率

}


def get_training_data(images, batch_size):
    indices = numpy.random.randint(len(images), size=batch_size) #随机抽取图像索引
    for i, index in enumerate(indices): # 遍历索引 处理每一张图
        image = images[index] # 取出图像
        image = random_transform(image, **random_transform_args) # 随机变换
        warped_img, target_img = random_warp(image) # 生成扭曲图像 和 对应的（未扭曲的）目标（原始）图像（核心操作）

        if i == 0: # 第一次循环时初始化数组
            # 创建与单张图像形状匹配的批量数组（batch_size × 高度 × 宽度 × 通道）
            warped_images = numpy.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = numpy.empty((batch_size,) + target_img.shape, warped_img.dtype)
        # 填充当前图像到批量数组中
        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images

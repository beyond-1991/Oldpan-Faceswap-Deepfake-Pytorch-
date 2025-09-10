import cv2
import numpy
import os


def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png")]


def load_images(image_paths, convert=None):
    # cv.imread 读取图像，再用cv2.resize统一调整为256*256，最后转换为numpy数组 BRG
    iter_all_images = (cv2.resize(cv2.imread(fn), (256, 256)) for fn in image_paths)
    if convert:# 将BRG转为RGB
        iter_all_images = (convert(img) for img in iter_all_images)
    for i, image in enumerate(iter_all_images):
        if i == 0: # 初始化数组
            # 创建与第一张图像形状匹配的批量数组（batch_size × 高度 × 宽度 × 通道）
            # 创建一个空的 numpy 数组 all_images，形状为 (图像总数, 256, 256, 3)（第一维是批量大小）。
            all_images = numpy.empty((len(image_paths),) + image.shape, dtype=image.dtype)
        #  遍历所有图像，依次存入all_images数组中

        all_images[i] = image
    return all_images #最终返回形状为 (N, 256, 256, 3) 的数组（N 是图像数量）

 
def get_transpose_axes(n): # n：输入数组的维度数量（例如批量图像的维度为 4：(N, H, W, C)，则 n=4）。
    """根据 n 的奇偶性，生成三组轴索引（y_axes、x_axes、[n-1]），
    核心是将原维度拆分为 “y 方向轴”“x 方向轴” 和 “最后一个轴”（通常是通道轴）。"""
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_images(images): # 拼接图像为大图像网格

    images_shape = numpy.array(images.shape) # 获取图像 eg:假设输入是形状为 (4, 256, 256, 3) 的4D数组（4张256×256的RGB图像） N H W C

    new_axes = get_transpose_axes(len(images_shape)) # 计算转置轴 eg：（以 n=4 为例，返回 ([1], [0], [3])）。
    new_shape = [numpy.prod(images_shape[x]) for x in new_axes]  # new_axes 为 ([1], [0], [3]) ，拼接后为 [1, 0, 3]
    #对每组轴的维度求乘积 对于4D数组，新形状为 [H*N, W, C] （将 N 个图像按行堆叠）。
    
   # 转置后数组形状变为 (256, 4, 256, 3)
    return numpy.transpose(
        images,
        axes=numpy.concatenate(new_axes)
    ).reshape(new_shape) # new_shape 为 [256*4, 256, 3] 即 (1024, 256, 3)


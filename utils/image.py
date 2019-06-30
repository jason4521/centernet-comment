import cv2
import numpy as np
import random

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_(image, mean, std):
    image -= mean
    image /= std

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)

# 把原图放在新图形的正中心，其余位置为 0 ，相当于对原图进行零填充
def crop_image(image, center, size):
    # 原图缩放后整张图的中心
    cty, ctx            = center
    # 需要 crop 区域的大小
    height, width       = size
    # 原图缩放后整张图的大小   [ H , W ]
    im_height, im_width = image.shape[0:2]
    # 占位：需要 crop 区域
    cropped_image       = np.zeros((height, width, 3), dtype=image.dtype)
    # 需要 crop 区域的左上角点（x0，y0），右下角点 （x1，y1）
    # 如果需要 crop 的图大于缩放后的原图，crop 后的图片即缩放的原图（保持不变）
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)
    # 中心点到需要 crop 区域四边的距离
    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty
    # 需要 crop 区域的中心点
    cropped_cty, cropped_ctx = height // 2, width // 2
    # slice() 函数实现切片对象，主要用在切片操作函数里的参数传递。
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset

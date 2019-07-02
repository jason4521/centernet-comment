import cv2
import math
import numpy as np
import torch
import random
import string

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop, draw_gaussian, gaussian_radius

from visdom import Visdom
vis = Visdom(env='train_centernet',port = 8099)

def _full_image_crop(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size   = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:, 0:4:2] += border[2]
    detections[:, 1:4:2] += border[0]
    return image, detections

def _resize_image(image, detections, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

def kp_detection(db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    # 每张图 最多 检测 128 个框
    max_tag_len = 128

    # allocating memory
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    ct_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    br_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    ct_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    tl_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    br_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    ct_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens    = np.zeros((batch_size, ), dtype=np.int32)

    # 训练集共多少张 118287
    db_size = db.db_inds.size
    # b_ind ，mini-batch中每张图片的索引，该函数每次返回一个 batch 的训练数据
    for b_ind in range(batch_size):
        # 默认进行 shuffle
        if not debug and k_ind == 0:
            db.shuffle_inds()

        # 从 shuffle 后的训练集选取 第 k_ind 张，返回图片在训练中的索引
        db_ind = db.db_inds[k_ind]
        # 每选取一张后 k_ind + 1 。由于训练多个 epoch ，k_ind 不能超过训练集大小
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        # reading detections
        # 读取标注数据 shape([[4,],1]) 前四个是角点坐标，加一个 类别
        detections = db.detections(db_ind)

        # 自己增加部分： 显示原图与标注数据
        img_raw = image.copy()
        # 添加框
        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            img_raw = cv2.rectangle(img_raw, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])),(112, 25, 25), 3)
            # 添加类别
            img_raw = cv2.putText(img_raw, str(category), (int(detection[0]), int(detection[1] + 25)),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        # 显示原图
        vis_img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        vis_img = np.transpose(vis_img_raw, (2, 0, 1))  # [H , W , C ]  -->  [C , H , W]
        vis.image(vis_img, win='raw_train_img', opts=dict(title='raw_train_img'))

        # 1. 数据增强
        # cropping an image randomly
        # 1.1 随机剪裁
        if not debug and rand_crop:
            # crop 产生的图片都是正方型
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)
        # 将不同大小的输入图片 转换成 [511,511]
        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        # 缩放比率( < 1 )      特征图大小 / 输入大小
        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        # 1.2 水平翻转
        if not debug and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        # 1.3 色彩变幻  更改 亮度，对比度，饱和度
        if not debug:
            # 变 0-1
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            # 均值方差均为 小于 1 的数
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))  # [H , W , C ]  -->  [C , H , W]

        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1
            #category = 0

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            xct, yct = (detection[2] + detection[0])/2., (detection[3]+detection[1])/2.

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)
            fxct = (xct * width_ratio)
            fyct = (yct * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)
            xct = int(fxct)
            yct = int(fyct)

            if gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad

                draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
                draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius, delte = 5)

            else:
                tl_heatmaps[b_ind, category, ytl, xtl] = 1
                br_heatmaps[b_ind, category, ybr, xbr] = 1
                ct_heatmaps[b_ind, category, yct, xct] = 1

            # 已经读取的框数
            tag_ind                      = tag_lens[b_ind]

            # 2.2 offset    (batch_size, 128, 2)   特征图角点 理论坐标 - 舍弃后坐标
            tl_regrs[b_ind, tag_ind, :]  = [fxtl - xtl, fytl - ytl]
            br_regrs[b_ind, tag_ind, :]  = [fxbr - xbr, fybr - ybr]
            ct_regrs[b_ind, tag_ind, :]  = [fxct - xct, fyct - yct]

            # 2.3 位置索引
            # xtl , ytl 特征图坐标
            # (batch_size, 128)  位置信息
            tl_tags[b_ind, tag_ind]      = ytl * output_size[1] + xtl
            br_tags[b_ind, tag_ind]      = ybr * output_size[1] + xbr
            ct_tags[b_ind, tag_ind]      = yct * output_size[1] + xct

            # shape(batch_size,)  每张图片有几个框，数值就是几
            tag_lens[b_ind]             += 1

    for b_ind in range(batch_size):
        # 表明 该 batch 内每张照片有几个框
        tag_len = tag_lens[b_ind]
        # onehot？     shape(batch_size, 128)
        tag_masks[b_ind, :tag_len] = 1

    images      = torch.from_numpy(images)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)     # (batch_size,80, 128, 128)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    ct_heatmaps = torch.from_numpy(ct_heatmaps)
    tl_regrs    = torch.from_numpy(tl_regrs)        # (batch_size, 128, 2)
    br_regrs    = torch.from_numpy(br_regrs)
    ct_regrs    = torch.from_numpy(ct_regrs)
    tl_tags     = torch.from_numpy(tl_tags)         # (batch_size, 128)
    br_tags     = torch.from_numpy(br_tags)
    ct_tags     = torch.from_numpy(ct_tags)
    tag_masks   = torch.from_numpy(tag_masks)

    # 自己增加部分： 显示处理后的图
    vis_img_augmentation = np.zeros((3,511,511))
    vis_img_augmentation[0] = images[0][-1,:,:]
    vis_img_augmentation[1] = images[0][1,:,:]
    vis_img_augmentation[2] = images[0][2,:,:]
    vis.image(vis_img_augmentation, win='img_augmentation', opts=dict(title='img_augmentation'))

    return {
        "xs": [images, tl_tags, br_tags, ct_tags],
        "ys": [tl_heatmaps, br_heatmaps, ct_heatmaps, tag_masks, tl_regrs, br_regrs, ct_regrs]
    }, k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)

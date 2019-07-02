# 简单说明  
本文是 `Kaiwen Duan` 开源的 `Centernet`代码的简单注释,理解错误之处请指教  

[源码地址：](https://github.com/Duankaiwen/CenterNet) /github.com/Duankaiwen/CenterNet     
[论文: ](https://arxiv.org/abs/1904.08189) CenterNet: Keypoint Triplets for Object Detection    

# Architecture

![Network_Structure](https://github.com/Dawning23/centernet-comment/blob/master/image/Network_Structure.jpg)  

# `/utils/image.py 中 crop_image()` 函数的作用   
将原图放置在 `inp_height = new_height | 127` 代码生成的新图的正中心，周围用 0 填充，如下图  

原图：  

![测试生成的图](https://github.com/Dawning23/centernet-comment/blob/master/image/raw_test_img.jpg)

函数处理过的图：  

![img_padding](https://github.com/Dawning23/centernet-comment/blob/master/image/img_padding.jpg)

# Centenet 测试流程  

1. 运行 `test.py` 函数最开始设置最基本的参数，`testing(db, nnet, result_dir, debug=debug)` 进入 `/test/coco.py` 文件的 `kp_detection()` 函数  


2. `kp_detection()` 函数流程：      

    ① 设置参数及测试集地址  
    ② 遍历测试集，每次循环读取1张照片  
    ③ 对照片进行缩放(默认保持不变)   
    ④ 缩放后，通过 `| 127`,零初始化一个更大的图，使用`crop_image()`函数将原图放置在初始化的图的正中心并归一化，转换成[C,H,W]    
    ⑤ 使用`decode_func()`函数检测原图与翻转后的图片，返回    
      ```
      # center = # torch.Size([2,70, 4])        1. 预测的中点坐标值  2.预测的类别值  3， 预测的置信度
      # detections = # torch.Size([2, 1000, 8]) 1. 预测的 bboxes 坐标 占 4  2. bboxes 对应的置信度  3. 角点置信度 占 2  4.预测的类别
      ```   
    ⑥ 转换翻转后的坐标 --> 翻转前  
    ⑦ 使用`_rescale_dets()`函数将生成的框与中心点对应到原图上  ，将出界框的置信度置为 -1 (padding原图引起的？)   
    ⑧ 按论文中 150 为分界，分成大框和小框，分别判定中心点是否在该框的中心区域，保留有中心点对应的框   
    ⑨ nms处理冗余的框，根据每张图 `max_per_image` 参数值删除置信度小的框  
    ⑩ 保存生成的数据  


3. `decode_func()`函数流程    

    该函数实际运行 `/models\py_utils\kp.py 文件 class kp(nn.Module): 的 _test(self, *xs, **kwargs) 部分`   

    1. 使用训练的网络提取特征，产生 8 个特征图：  

        ```   
        tl_heat
        br_heat
        ct_heat     # torch.Size([1, 80, 128, 128])
        tl_tag      # torch.Size([1, 128, 1])
        br_tag      # embed
        tl_regr     # offset
        br_regr     # torch.Size([1, 128, 2])
        ct_regr     # torch.Size([1, 128, 2])
        ```   

    2. 使用 `_decode()` 函数，流程如下：    

        ① 首先对三个热图进行 `torch.sigmoid()` ，只保留 3*3 的最大值池化中的最大值，其余均设置为 0     
        ② 提取 heatmap 中值高的前 70 个点的坐标及每个值的位置索引和所属类别   
        ③ 根据②的索引，对应位置的坐标标加 offset --> 特征图理论坐标    
        ④ 70个角点，产生 4900 个框的可能性，将其中 符合条件：[角点的 embed 距离 大于 dist，两个角点 不是相同类，反向框] 的框置信度设置为 -1    
        ⑤ 根据置信度排名，每张图选取前 1000 个框   
        ⑥ 返回

          ```  
          # center = # torch.Size([2,70, 4])        1. 预测的中点坐标值  2.预测的类别值  3， 预测的置信度
          # detections = # torch.Size([2, 1000, 8]) 1. 预测的 bboxes 坐标 占 4  2. bboxes 对应的置信度  3. 角点置信度 占 2  4.预测的类别
          ```  

# Centenet 训练流程

1. `sample\coco.py kp_detection()` 函数每次读取一个 `batch` 的训练集照片及其 位置索引 embed offset heatmap 流程如下：   
  ① 初始化一堆参数，设置每张图最多检测 128 个框，占位所有的输出  
  ② 训练集 shuffle 一次，每次读取一张照片到 batch 中  
  ③ 读取该照片的 lable 数据，对照片进行数据增强包括 [随机剪裁，水平翻转，色彩变幻]
  ④ 利用标注数据在角点及中电附近画高斯分布的热图(heatmaps) ，计算点在特征图与原图的偏移(regrs) ，确定点在特征图的位置索引(tags) ，该 batch 内每张照片有几个框(tag_masks)-有几个框前边就是几个 1   


2. `for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):` 每次将生成的 batch 送入 网络训练，更新参数  


3. `/models\py_utils\kp.py 文件 class kp(nn.Module): 的 _train(self, *xs): 部分` 计算输出   

  ![网络结构图](https://github.com/Dawning23/centernet-comment/blob/master/image/Backbone_Structure.JPG)


4. `/models\py_utils\kp.py 文件 class AELoss(nn.Module): 部分` 计算损失  

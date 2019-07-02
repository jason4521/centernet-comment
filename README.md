# 简单说明  
本文是 `Kaiwen Duan` 开源的 `Centernet`代码额简单注释,理解错误之处请指教  
[源码地址：](https://github.com/Duankaiwen/CenterNet)   
[论文: ](https://arxiv.org/abs/1904.08189)  

# Architecture

![Network_Structure](https://github.com/Dawning23/centernet-comment/blob/master/Network_Structure.jpg)  

# 注释中的 `crop_image()` 函数的作用   
将原图放置在 `inp_height = new_height | 127` 代码生成的新图的正中心，周围用 0 填充，如下图  

原图：  

![Network_Structure](https://github.com/Dawning23/centernet-comment/blob/master/raw_test_img.jpg)

函数处理过的图：  

![Network_Structure](https://github.com/Dawning23/centernet-comment/blob/master/img_padding.jpg)

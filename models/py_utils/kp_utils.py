import pdb
import torch
import torch.nn as nn

from .utils import convolution, residual

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

def make_tl_layer(dim):
    return None

def make_br_layer(dim):
    return None

def make_ct_layer(dim):
    return None

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

# ct_regr = [batch_size , 16384 , 2] , ct_embed [batch_size , 128]
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    # 1. ind 变成 [batch_size , 128 ，1 ]  在位置2 加一维度
    # temp  = ind.unsqueeze(2)
    # 2. [batch_size , 128 ，2 ]  复制
    # temp = temp.expand(ind.size(0), ind.size(1), dim)

    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)  # 根据索引重组 torch.Size([1, 128, 2])
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask] # torch.Size([1, 128, 2])
        feat = feat.view(-1, dim)
    return feat

# heat = torch.Size([2, 80, 128, 192])
# 只保留 3*3 pooling 中最大值
def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2 # k = 3

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)   # torch.Size([2, 80, 128, 192])
    keep = (hmax == heat).float()      # 相同位置值相同为1不同为0
    return heat * keep  # 只保留 3*3 pooling 中最大值 --> 对应高斯分布？ 只保留最大值那个角点

# ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)   # torch.Size([1, 128, 2])
# ct_regr = torch.Size([1,  2, 128, 128])   , ct_inds = # (batch_size, 128) 真实的 embed
def _tranpose_and_gather_feat(feat, ind):
    # ct_regr =  [batch_size , 128 ,128, 2]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3)) # ct_regr = [batch_size , 16384 , 2]
    feat = _gather_feat(feat, ind)
    return feat
# scores = torch.Size([2, 80, 128, 192])   K:70
def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    # 有序排列所有 heatmap 中的值
    # temp = scores.view(batch, -1)   # torch.Size([2, 1966080])
    # topk_scores, topk_inds = [2,70]   进一步选出 heatmap 中前 70 大的值，及其索引
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    # 具体是哪个类 0-80
    topk_clses = (topk_inds / (height * width)).int()
    # 在该类特征图上的具体位置索引
    topk_inds = topk_inds % (height * width)
    # 计算当前角点特征图上的角点坐标
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    # 1. 每张图前 70 个 角点置信度最高的值   2. 在一张一维特征图上的具体位置索引   3. 选出的 70 个值对应的类  4. 选出的值在特征图上的坐标
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, ct_heat, ct_regr, 
    K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)    # 例子： torch.Size([2, 80, 128, 192])
    br_heat = torch.sigmoid(br_heat)
    ct_heat = torch.sigmoid(ct_heat)

    # perform nms on heatmaps
    # 只保留 [kernel,kernel](3*3) 区域内最大值那个角点，其余赋值 0
    # 例子： torch.Size([2, 80, 128, 192])
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)
    ct_heat = _nms(ct_heat, kernel=kernel)

    # 提取 heatmap 中值高的前 70 个值 -->  1. 每张图前 70 个 角点置信度最高的值   2. 在一张一维特征图上的具体位置索引   3. 选出的 70 个值对应的类  4. 选出的值在特征图上的坐标
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)
    # 同上 K = 70   70个左上角 70个右下角 共 4900 个框
    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)     # torch.Size([2, 70, 70])
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
    ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        # 计算 offset            tl_regr = torch.Size([2, 2, 128, 192])
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)   # torch.Size([2, 70, 2])
        tl_regr = tl_regr.view(batch, K, 1, 2)      # torch.Size([2, 70, 1, 2])
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)
        ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)
        ct_regr = ct_regr.view(batch, 1, K, 2)

        # temp = tl_regr[..., 0]  torch.Size([2, 70, 1])
        # 特征图判断的坐标加 offset --> 特征图理论坐标
        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
        ct_xs = ct_xs + ct_regr[..., 0]
        ct_ys = ct_ys + ct_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    # 不考虑 类别  每张图产生 4900 个框
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    # 计算 角点的 embed 距离 dist
    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists  = torch.abs(tl_tag - br_tag)

    # 角点的 置信度 均值
    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    # 70 个框的置信度均值
    scores    = (tl_scores + br_scores) / 2      #  torch.Size([2, 70, 70])
    # reject boxes based on classes
    # 两个角点 不是相同类 的索引 cls_inds
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    # 反向框
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)
    # 冗余的框置信度设置为 -1
    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1
    scores = scores.view(batch, -1)  # torch.Size([2, 4900])
    # 选取置信度前 1000 的框
    scores, inds = torch.topk(scores, num_dets)  # torch.Size([2, 1000])
    scores = scores.unsqueeze(2)    # torch.Size([2, 1000, 1])
    bboxes = bboxes.view(batch, -1, 4)  # torch.Size([2, 70, 70, 4]) --> torch.Size([2, 4900, 4])
    # 根据置信度前 1000 框的索引提取该框的坐标 torch.Size([2, 1000, 4])
    bboxes = _gather_feat(bboxes, inds)
    
    #width = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    #height = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    # 根据置信度前 1000 框的索引提取该框的类别 torch.Size([2, 1000, 1])
    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    # 根据置信度前 1000 框的索引提取该角点的置信度 torch.Size([2, 1000, 1])
    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    ct_xs = ct_xs[:,0,:]
    ct_ys = ct_ys[:,0,:]

    # center = 1.预测的中点坐标值  2.预测的类别值  3， 预测的置信度
    center = torch.cat([ct_xs.unsqueeze(2), ct_ys.unsqueeze(2), ct_clses.float().unsqueeze(2), ct_scores.unsqueeze(2)], dim=2)  # torch.Size([2,70, 4])
    # detections = 1. 预测的 bboxes 坐标  2. bboxes 对应的置信度  3. 角点置信度  4.预测的类别
    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)    # torch.Size([2, 1000, 8])  bboxes 占 4
    return detections, center

def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        # 公式见 cornernet P5
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float() # 框的个数
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    if len(tag_mean.size()) < 2:
        tag_mean = tag_mean.unsqueeze(0)
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

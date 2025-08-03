---
title: YOLOV8系列模型部署与后处理
date: 2025-08-03 19:25:27
categories: 
  - "深度学习"
  - "模型部署"
tags:
  - YOLOv8
---

# YOLOv8系列模型后处理方案

YOLOv8系列模型是如今工业界使用较多的一些模型, 本文详细描述各类型的输出结构和后处理方案细节。采取的部署方案更偏向边缘端，例如RKNN等，从检测头开始输出，不使用Ultralytics这种常规结构: `[1, 4 + num_classes, N]`的后处理方案。

* 目标检测 (YOLOv8)
* 旋转检测 (YOLOv8-OBB)
* 关键点 (YOLOv8-Pose)
* 实例分割 (YOLOv8-Seg)
  
## 模型部署基本流程

模型推理大都经过以下过程

1. 预处理Resize、Normalize等， 例如一些Letterbox、归一化(通常为除以255)等，组织图片为需要的输入数据
2. 模型推理，可以是ONNX、RKNN、TensorRT等模型，API有所不同、输入也有可能不同(主要是NCHW与NHWC之间的区别)，就是把步骤1获得的图片输入进模型，得到输出。
3. 输出后处理，对模型输出进行进一步处理、例如DFL、NMS等内容（一些模型可以通过量化、算子合并与拆分来实现模型的精度与速度平衡，会涉及后处理的变动）

## 一、YOLOv8

### 输入输出

部署的输入输出结构如下:

![YOLOv8ONNX](/images/YOLOV8/yolov8det.png)

以上模型输入尺寸为[1, 3, 1260, 1260], stride为[8, 16, 32]，模型有两个检测类别，三个检测头输出为[1, 66, 160, 160]，[1, 66, 80, 80]，[1, 66, 40, 40]， 每个stride尺寸方格内进行预测，假设stride为8，则1260x1260的尺寸内共有160x160个预测方格。前64组数据用于预测框，后面两个数据表示预测类别置信度，(这个模型中将类别置信度和坐标拼在一起，也可使用[RKNN部署方案](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8)，后处理需要进行一定定的更改), 设定的regmax为16，因此用于计算框的数据长度为16x4为64。

### 后处理

#### 1. DFL框解码

regmax为16，分为4组，分别预测l、t、b、r分别表示左上的x、y与右下的x、y坐标。首先对16个一组的坐标值进行计算Softmax，与regmax的16个索引值相乘后相加，得到最终的坐标值(这里计算的是一个期望值)。得到输出后，根据中心点计算坐标回归，这里的中心点是每个预测方格的中心点，以stride为8时举例，第一个中心点为(4, 4), 根据这个中心点可以求出这个方格中的预测坐标x1, y1, x2, y2. 代码如下。


```
    C, H, W = output.shape                           # [66, 160, 160]
    output = output.reshape(C, -1).T                 # [160x160, 66]                                 
    bbox_dfl = output[:, :NUM_BINS * 4].reshape(-1, 4, NUM_BINS)  #[160x160, 4, 16]

    prob = softmax(bbox_dfl, axis=-1)                  
    proj = np.arange(NUM_BINS, dtype=np.float32)      # [0, 1, 2, ... 15]        
    bbox = np.sum(prob * proj, axis=-1) * stride      # [160x160, 4]

    y, x = np.divmod(np.arange(H * W), W)
    grid_x = (x * stride + stride / 2)
    grid_y = (y * stride + stride / 2)                # 构造中心点坐标

    x1 = (grid_x - bbox[:, 0])                        # 坐标回归
    y1 = (grid_y - bbox[:, 1])
    x2 = (grid_x + bbox[:, 2])
    y2 = (grid_y + bbox[:, 3])
```

#### 2. 类别预测

类别预测较为简单，对每个预测类别输出做sigmoid即可, 即可得到最终的置信度。 (部分网络可能使用Softmax计算，用于输出不可能同时满足两个类别的情况)，类别大于conf_thresh阈值需要使用NMS进行进一步过滤。

```
    cls_logits = output[:, NUM_BINS * 4:]
    probs = sigmoid(cls_logits)
```

#### 3. NMS

做NMS是为了过滤一些预测重复的框，选择置信度最高的框。判断标准就是计算IOU，IOU大于阈值的框被认为是重复框，只保留置信度最高的框。![IOU](/images/YOLOV8/iou.png)

```
def nms_boxes(boxes, scores, nms_thresh=0.5):
    x1, y1, x2, y2 = boxes.T               # 坐标解包，x1为一连串的坐标
    areas = (x2 - x1) * (y2 - y1)          # 计算原有框的面积
    order = scores.argsort()[::-1]         # 排序，order这里是索引，置信度大的排在前面
    keep = []                              # 返回过滤后框索引

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= nms_thresh)[0] + 1]
    return np.array(keep, dtype=np.int32)
```

先计算相交部分面积Inner，需要找到相交部分的左上角坐标[xx1， yy1]和右下角坐标[xx2, yy2]。假设原来的两个框坐标分别为[x1, y1, x2, y2],[x1', y1', x2', y2'], xx1与yy1是x1与x1'和y1与y1'比较的较大的坐标，xx2与yy2相反， 计算Inter时需要考虑不相交的情况。相并的部分面积就是两个框之和减去inter面积，计算IOU之后即可过滤IOU大于nms_thresh的部分，这里会先保留score较大的框。

#### 4. 其他处理
这里需要处理一下坐标回归和数据组织形式，图片输入到模型是经过Resize的，需要将坐标回归到原图的坐标


## 二、YOLOv8-OBB (旋转检测)

### 输入输出

部署的输入输出结构如下:

![YOLOv8OBBONNX](/images/YOLOV8/yolov8obb.png)

以上模型输入尺寸为[1, 3, 1260, 1260], stride为[8, 16, 32]，模型有两个检测类别，三个检测头输出为[1, 67, 160, 160]，[1, 67, 80, 80]，[1, 67, 40, 40]，这个模型中我将角度拼到了预测框与分类值后面。

### 后处理

#### 1. 角度计算

框计算与类别预测均与YOLOv8一致，角度计算非常简单，根据角度的输出计算真实角度即可，代码如下。这里theta这样计算是为了回归上的稳定，具体原理可参考相关论文

```
theta= output[num_box_channels, :]       # (N,)
theta = (sigmoid(theta) - 0.25) * np.pi  # [-pi/4, 3pi/4] in radians

```

### 2. RotatedNMS
不同于YOLOv8，obb在计算IOU时需要考虑旋转角度，这里需要使用旋转框的IOU计算方式。
```
def rotated_iou_numpy(box1, box2):
    """
    计算两个旋转框的 IoU，box 为 [cx, cy, w, h, theta]，theta 单位为角度
    """
    rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

    int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    if int_pts is None or len(int_pts) < 3:
        return 0.0

    inter_area = cv2.contourArea(int_pts)
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def rotated_nms(boxes, scores, iou_thresh=0.1):
    """
    NumPy 实现的旋转框 NMS
    boxes: (N, 5) -> cx, cy, w, h, angle
    scores: (N,)
    返回保留索引
    """
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = np.array([
            rotated_iou_numpy(boxes[i], boxes[j])
            for j in order[1:]
        ])
        order = order[1:][ious <= iou_thresh]
    return np.array(keep, dtype=np.int32)
```
计算IOU时相较于普通的矩形框，需要使用旋转框的IOU计算方式，内部实现需要一些空间计算相关知识，不详细展开。Python部署可以使用Shaply、C++可以使用OpenCV，

#### 3. 其他处理
最后需要将坐标回归到原图的坐标，由于是旋转框，最终输出一般为中心点+宽高+角度。输出时可以进行Regularize，这里搬运Ultralytics中的实现。将角度限制至[0, pi/2]范围，实测好像在OpenCV画图下作用不大，负角度也能绘制，在其他数据处理可以用上
```
def regularize_rboxes(rboxes):
    """
    Regularize rotated bounding boxes to range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input rotated boxes with shape (N, 5) in xywhr format.

    Returns:
        (torch.Tensor): Regularized rotated boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes
```
## 三、YOLOv8-Pose (关键点检测)

后续补充...
### 输入输出

### 后处理

#### 1. 关键点解码

#### 2. 其他处理



## 四、YOLOv8-Seg (实体分割)

后续补充...
### 输入输出

部署的输入输出结构如下:

### 后处理

#### 1. mask输出计算


#### 2. 其他处理


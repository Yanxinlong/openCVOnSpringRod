import torch
import cv2
import time

# post_processing这个python函数的功能是对预测得到的框做后期处理
# (1)对于太接近左右边界的预测框，忽略它，防止它降低检测的准确度
# (2)建立简单的排错机制：如果一张图像中能检测到trip，理论上不可能再检测出no_trip，所以删除检测到的no_trip；对bolt做同样的处理
# det_bbox :tensor([[9.60000e+01, 3.25000e+02, 1.64000e+02, 4.50000e+02, 8.47130e-02, 1.00000e+00],
#                  [4.08000e+02, 5.20000e+01, 4.93000e+02, 1.47000e+02, 5.33061e-02, 1.00000e+00],
#                  [6.60000e+01, 5.40000e+01, 1.34000e+02, 1.48000e+02, 5.01959e-02, 1.00000e+00]], device='cuda:0')
def post_processing(det_bbox,x_min,x_max):
    # 输入det_bbox是GPU tensor,如上所示
    classes = []                             # 用于记录检测到的不同类别的目标，0-trip,1-bolt,2-no_trip,3-no_bolt
    indexs = []                              # 用于记录不满足x坐标范围的预测框的索引（在tensor中的索引，第几行）
    for i, det in enumerate(det_bbox):
        if (det[0]<x_min) or (det[2]>x_max):
            indexs.append(i)

    for j in indexs:
        det_bbox = det_bbox[torch.arange(det_bbox.size(0))!=j]  # 根据预测框x坐标判断它是否在合理的位置，如果太靠近左右边界，根据索引删除这个预测框信息

    for *xyxy, conf, cls in det_bbox:
        cls = cls.cpu()
        cls = int(cls)
        classes.append(cls)

    if (0 in classes) and (1 in classes):
        classes.remove(1)

    return det_bbox,classes

 

def save_normal(image,camera_id):
    # 输入image是ndarray-3维数组
    if camera_id == 0:
        path = "record/Camera_0/"                              # 保存camera-0检测到的异常图像的文件夹
    if camera_id == 1:
        path = "record/Camera_1/"                   

    name = time.strftime("%m-%d-%H-%M-%S", time.localtime())   # 如 08-04 11:32:49 , 字符串类型

    cv2.imwrite(path+name+".png",image)







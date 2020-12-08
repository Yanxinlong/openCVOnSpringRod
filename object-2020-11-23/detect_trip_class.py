# coding: utf-8
'''
Created time: 2020-11-23
Author: chenzhiwen 
Function: 用于检测弹条是W型弹条还是三型弹条
'''

import argparse
from models import *  
from utils.datasets import *
from utils.utils import *
from tools import post_processing

#==========设置常量和超参数
cfg = "cfg/yolov3-tiny-1.cfg"                      
weights = "weights/last-1.pt"       
img_size = 512
device = torch_utils.select_device(device='0')

parser = argparse.ArgumentParser()
parser.add_argument('--names', type=str, default='data/data-1.names', help='*.names path')
parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')                 # 根据实际情况设置
parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')                        # 根据实际情况设置
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()
opt.cfg = check_file(cfg)  
opt.names = check_file(opt.names)  


def get_model_trip(cfg,img_size,weights,device):
#=====模型初始化
    model = Darknet(cfg, img_size)
#=====加载参数
    attempt_download(weights)
    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        load_darknet_weights(model, weights)
    model.to(device).eval()
    return model


def detect(model,image,image_tensor): 
    view_img = False
    im0 = image                                # 原图--(H,W,3) ，比如（1024,1280,3）
    im0 = cv2.resize(im0,(640,512))            # 原图可视化大小 ，比如 (512,640)-（H,W)
    img = image_tensor                         # 图像tensor--(1,3,H,W) ，比如（1,3,488,512）

#============推理========================
    pred = model(img, augment=opt.augment)[0]                     # ([1, 4500, 9])
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)    
    det = pred[0]                                                                       # 它是1个[n,6]的tensor,n表示预测框的数目 

    record = []
    if det is not None and len(det):                                                    # len(det)就是预测框的数目       
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()         # 从img_size到im0的尺寸rescale boxes
        det,classes = post_processing(det,x_min=20,x_max=620)                           # 对检测框做后期处理,classes表示处理后得到的目标类别
        record = classes
    
    return record


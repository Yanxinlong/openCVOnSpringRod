import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from tools import post_processing

#==========设置常量和超参数
cfg = "cfg/yolov3-tiny.cfg"                      
weights = "weights/last-w.pt"
img_size = 512
device = torch_utils.select_device(device='0')

parser = argparse.ArgumentParser()
parser.add_argument('--names', type=str, default='data/data.names', help='*.names path')
parser.add_argument('--conf-thres', type=float, default=0.26, help='object confidence threshold')                 # 根据实际情况设置
parser.add_argument('--iou-thres', type=float, default=0.31, help='IOU threshold for NMS')                        # 根据实际情况设置
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()
opt.cfg = check_file(cfg)  
opt.names = check_file(opt.names)  


def get_model_1(cfg,img_size,weights,device):
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


def detect_1(model,image,image_tensor): 
    view_img = True
    im0 = image                                # 原图--(H,W,3) ，比如（1024,1280,3）
    im0 = cv2.resize(im0,(640,512))            # 原图可视化大小 ，比如 (512,640)
    img = image_tensor                         # 图像tensor--(1,3,H,W) ，比如（1,3,488,512）

#==========获取类别名和调色板==========
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

#============推理========================
    pred = model(img, augment=opt.augment)[0]                     # ([1, 4500, 9])

    # 列表元素是gpu tensor=[x1, y1, x2, y2, conf, cls]，里面只有一个元素
    # 如，[tensor([[143.77414,  91.26861, 243.92767, 168.44662,   0.27712,   1.00000]], device='cuda:0')]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)    
    det = pred[0]                                                                       # 它是1个[n,6]的tensor,n表示预测框的数目 
  
    record = []
    if det is not None and len(det):                                                    # len(det)就是预测框的数目       
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()         # 从img_size到im0的尺寸rescale boxes

        det,classes = post_processing(det,x_min=20,x_max=620)                                              # 对检测框做后期处理,classes表示处理后得到的目标类别
        record = classes
    
        for *xyxy, conf, cls in det:                  
# xyxy, conf, cls是一个预测框的信息
# xyxy: [tensor(513., device='cuda:0'), tensor(61., device='cuda:0'), tensor(580., device='cuda:0'), tensor(137., device='cuda:0')]
# conf: tensor(0.08830, device='cuda:0')
# cls: tensor(1., device='cuda:0')
            if view_img:  
                label = '%s %.2f' % (names[int(cls)], conf)                               # 将bbox加到原图像
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

    # 显示检测结果
    if view_img:
        cv2.imshow("Camera_1 Detect Result",im0)
        if cv2.waitKey(1) == ord('q'):  
            raise StopIteration

    return record



# 推理方式：
# model = get_model(cfg,img_size,weights,device)       
# with torch.no_grad():
#         detect(model,image,image_tensor)

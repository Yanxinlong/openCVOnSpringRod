# 运行代码注意事项
1. 注意修改摄像机的COM端口号
2. 在detect_0，detect_1选择weights中的权重，选择E型或者W型弹条模型
```python
weights = "weights/last-w.pt"
```
3. 在detect_0，detect_1 调整识别率
```python
# 根据实际情况设置
parser.add_argument('--conf-thres', type=float, default=0.26, help='object confidence threshold')       
parser.add_argument('--iou-thres', type=float, default=0.31, help='IOU threshold for NMS')  
```
4. 分别运行inference_0和inference_1启动小车两侧识别程序

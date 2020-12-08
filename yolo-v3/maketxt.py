import os
import random
 
val_percent = 0.1
train_percent = 0.9
xmlfilepath = 'data_v2/labels'                # 标签为xml或txt
txtsavepath = 'data_v2/ImageSets'
total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
list = range(num)
tv = int(num * val_percent)
tr = int(tv * train_percent)
val = random.sample(list, tv)
train = random.sample(val, tr)

ftrain = open('data_v2/ImageSets/train.txt', 'w')
fval = open('data_v2/ImageSets/val.txt', 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in val:
        fval.write(name)
    else:
        ftrain.write(name)

ftrain.close()
fval.close()
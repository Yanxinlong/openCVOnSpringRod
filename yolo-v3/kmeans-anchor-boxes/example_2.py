import glob
import numpy as np
from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "../data_v2/labels"
CLUSTERS = 6

def load_dataset(path):
	dataset = []
	for txt_file in glob.glob("{}/*txt".format(ANNOTATIONS_PATH)):
		fo = open(txt_file, "r")
		for line in fo.readlines():   
			line = line.strip()                        
			line = line.split(" ")          
			width = float(line[3])
			height = float(line[4])

			dataset.append([width,height])

	return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS)
print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100)+"\n")
print("Boxes: {}".format(out*512))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
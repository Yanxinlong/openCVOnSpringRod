import pickle
import os
from os import listdir, getcwd
from os.path import join
 
sets = ['train','val']

for image_set in sets:
    image_ids = open('./data_v2/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('./data_v2/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('data_v2/images/%s.jpeg\n' % (image_id))
    list_file.close()
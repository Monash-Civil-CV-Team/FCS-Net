from model import *
import os.path
import Seg_test
from joint_for_fullscale_1 import joint_for_fullscale as joi
import math
from Cal_IoU import iou
from cal_resize_iou import cal_resize_iou
import cv2 as cv
import math
#调整原图尺寸，裁剪成patch
crop_size=512
crop_dir='new_data/test/crack/mask/'
in_dir='binary/bw/'
def cut_all_img(in_dir,  crop_dir):
    os.makedirs(in_dir,exist_ok=True)
    os.makedirs(crop_dir,exist_ok=True)
    for file in os.listdir(in_dir):
          file_path=os.path.join(in_dir, file)
          print('HHHA:0===>', file_path)
          try:
              img = cv.imread(file_path)
              row=math.floor(img.shape[0]/crop_size)
              col=math.floor(img.shape[1]/crop_size)
              resized = cv.resize(img, ((crop_size*col),(crop_size*row)))
              #for i in range(6):
              for i in range(row):
                  #for j in range(9):
                  for j in range(col):
                      cropped = resized[crop_size * i:crop_size * (i + 1), crop_size * j:crop_size * (j + 1)]
                      #cv.imwrite(crop_dir + file + '%d_%d.jpg' % (i, j), cropped)   file.split['.'][0]+'_'
                      #cv.imwrite(crop_dir + file + '%d_%d.jpg' % (i, j), cropped)
                      cv.imwrite(crop_dir + file.split('.')[0]+'_' + '%d_%d.jpg' % (i, j), cropped)
          except:
              print('failed to resize：', file_path)

#cut_all_img(in_dir='new_test_set/', crop_dir='new_data/test/crack/image/') #in_dir是全尺寸原图地址，out_dir是裁剪patch的保存地址，也就是crack_input_dir
cut_all_img(in_dir, crop_dir)

from model import *
import os.path
import Seg_test
from joint_for_fullscale import joint_for_fullscale as joi
from joint_for_fullscale import joint_by_slide_window as joi_SW
from Cal_IoU import iou
from cal_resize_iou import cal_resize_iou
import cv2 as cv

# --------------------------------------------------test configuration--------------------------------------------
model_name = 'originalresnet' #model name
model = FCS_Net()        #model strucutre
joint_with_SW = False
#joint_with_SW = True
save_dir = 'new_data/test/test results/'+model_name+ '/'     # results output direction
os.makedirs(save_dir)
#crack_input_dir = 'new_data/test_SW/crack/image/'     #image input direction
crack_input_dir = 'new_data/test/crack/image/'     #image input direction
#crack_input_dir = 'new_data/train/928crack/image/'     #image input direction
crack_save_dir = save_dir + 'crack/'                    #crack image output direction
os.makedirs(crack_save_dir)
#background_input_dir = 'new_data/test_SW/background/image/'    #background image output direction
background_input_dir = 'new_data/test/background/image/'    #background image output direction
#background_input_dir = 'new_data/train/background/image/'    #background image output direction
background_save_dir = save_dir + 'background/'
os.mkdir(background_save_dir)
#crack_label_dir = 'new_data/test_SW/crack/mask/'
crack_label_dir = 'new_data/test/crack/mask/'
fullscale_label_dir = 'new_data/test/fullscale_mask/'
fullscale_save_dir = 'new_data/test/test results/'+model_name+'/fullscale/'

#crack_label_dir = 'new_data/train/928crack/mask/'
#fullscale_label_dir = 'new_data/test/fullscale_mask/'
#fullscale_save_dir = 'new_data/test/test results/'+model_name+'/fullscale/'

# --------------------------------------------------test----------------------------------------------------
Seg_test.test_for_batch(model_name = model_name, model = model,
                        test_input_dir = crack_input_dir, test_save_dir = crack_save_dir)
Seg_test.test_for_batch(model_name = model_name, model = model,
                        test_input_dir = background_input_dir, test_save_dir = background_save_dir)

# joint for full-scale
if joint_with_SW:
    joi_SW(model_name=model_name)
else:
    joi(model_name=model_name)

# calculate IoU for single images and fullscale images


iou(label_dir=crack_label_dir, pred_dir= crack_save_dir, model_name=model_name, xls_dir= save_dir, type='crack')
iou(label_dir=fullscale_label_dir, pred_dir= fullscale_save_dir, model_name=model_name, xls_dir= save_dir, type='fullscale')

cal_resize_iou(ori_mask_path='new_data/test/test_original_mask/',
               pred_mask_path=fullscale_save_dir,
               write_xlsx_path=save_dir,
               model_name=model_name)



# color inverse
crack_inverse_save = save_dir+'inverse_crack/'
os.makedirs(crack_inverse_save)
fullscale_inverse_save = save_dir+'inverse_fullscale/'
os.makedirs(fullscale_inverse_save)

for file in os.listdir(crack_save_dir):
    img = np.array(cv.imread(crack_save_dir+file, -1))
    inverse = (255-img)
    inverse[inverse>125] = 255
    inverse[inverse<=125] = 0
    cv.imwrite(crack_inverse_save+file, inverse)


for file in os.listdir(fullscale_save_dir):
    img = np.array(cv.imread(fullscale_save_dir + file, -1))
    inverse = (255 - img)
    inverse[inverse > 125] = 255
    inverse[inverse <= 125] = 0
    cv.imwrite(fullscale_inverse_save + file, inverse)




import os.path
import PIL.Image as Image
from shutil import copyfile
import cv2 as cv
import numpy as np

def joint_for_fullscale(model_name):
    crack_dir = 'new_data/test/test results/'+model_name+'/crack/'
    background_dir = 'new_data/test/test results/'+model_name+'/background/'

    total_dir = 'new_data/test/test results/'+model_name+'/total/'
    os.makedirs(total_dir)

    fullscale_dir='new_data/test/test results/'+model_name+'/fullscale/'
    os.makedirs(fullscale_dir)

    img108_dir = 'new_data/test/test results/'+model_name+'/108/'
    os.makedirs(img108_dir)

    for crack in os.listdir(crack_dir):
        if '108' in crack.split('.')[0]:
            copyfile(crack_dir+crack, img108_dir+crack)
        else:
            copyfile(crack_dir+crack, total_dir+crack)

    for background in os.listdir(background_dir):
        if '108' in background.split('.')[0]:
            copyfile(background_dir+background, img108_dir+background)
        else:
            copyfile(background_dir+background, total_dir+background)

    def joint(total_dir, fullscale_dir, img108_dir=None):
        img_format = ['.jpg', '.JPG'] ##图片格式
        img_size = 512 ##每张图片的大小
        img_row=6 ##图片间隔，也就是合并成一张图后，一共有几行
        img_col = 9 ##有几列
        ##定义图像的拼接函数
        def common_compose():
            img_names = [name for name in os.listdir(total_dir) for item in img_format if os.path.splitext(name)[1] == item]
            to_img = Image.new('L', (img_col*img_size, img_row*img_size)) #创建一个新图
             ##循环遍历，把每张图片按顺序粘贴到对应位置上
            for i in range(0, 2106, 54):
              for y in range(0, img_row):
                  for x in range(0, img_col):
                      from_img = Image.open(total_dir+img_names[img_col*(y)+x+i]).resize((img_size, img_size), Image.ANTIALIAS)
                      to_img.paste(from_img, ((x)*img_size, (y)*img_size))
              to_img.save(fullscale_dir+img_names[i][0:3]+'.jpg') ##保存新图
        common_compose()

        def special_compose():
            img_names = [name for name in os.listdir(img108_dir)]
            to_img = Image.new('L', (10*img_size, 7*img_size))
            for y in range(0, 7):
                for x in range(0, 10):
                    from_img = Image.open(img108_dir+img_names[10*y+x]).resize((img_size, img_size), Image.ANTIALIAS)
                    to_img.paste(from_img, (x*512, y*512))
            to_img.save(fullscale_dir+img_names[0][0:3]+'.jpg')
        special_compose()
    joint(total_dir, fullscale_dir,img108_dir=img108_dir)
    return
#img.paste(image, box): 将一张图片粘贴到另一张图像上，变量box为给定左上角的2元组，或者定义整个板块的上下左右坐标的4元组，或者为空（默认为（0，0））

def element_wise_merge(ori_mask, sub_mask, x, y):
    #x and y are the upper left coordinate of sub_mask, corresponding to the row and column number respectively
    for i in range(512):
        for j in range(512):
            row = x+i
            column = y+j
            ori_mask[row, column] = max(ori_mask[row, column], sub_mask[i, j])

def joint_by_slide_window(model_name):
    crack_dir = 'new_data/test/test results/'+model_name+'/crack/'
    crack_lists = os.listdir(crack_dir)

    background_dir = 'new_data/test/test results/'+model_name+'/background/'
    background_lists = os.listdir(background_dir)

    fullscale_dir='new_data/test/fullscale_mask/'
    ori_mask_dir = 'new_data/test/test_original_mask/'
    fullscale_lists = os.listdir(fullscale_dir)

    merge_dir = 'new_data/test/test results/'+model_name+'/fullscale/'
    os.makedirs(merge_dir)

    for fullscale_img in fullscale_lists:
        fullscale = cv.imread(ori_mask_dir+fullscale_img.split('.')[0]+'.png')
        ori_mask = np.zeros((fullscale.shape[0], fullscale.shape[1]))
        for crack_img in crack_lists:
            if crack_img.split('_')[0] == fullscale_img.split('.')[0]:
                crack = cv.imread(crack_dir+crack_img, -1)
                thresh, bi = cv.threshold(crack, 0, 255, cv.THRESH_BINARY)
                x = int(crack_img.split('_')[1])
                y = int((crack_img.split('_')[2]).split('.')[0])
                element_wise_merge(ori_mask, bi, x, y)
        for background_img in background_lists:
            if background_img.split('_')[0] == fullscale_img.split('.')[0]:
                background = cv.imread(background_dir + background_img, -1)
                thresh, bi = cv.threshold(background, 0, 255, cv.THRESH_BINARY)
                x = int(background_img.split('_')[1])
                y = int((background_img.split('_')[2]).split('.')[0])
                element_wise_merge(ori_mask, bi, x, y)
        cv.imwrite(merge_dir+fullscale_img, ori_mask*255)





import io
import imageio
from model import *
import skimage.transform as trans
import tensorflow as tf
import cv2 as cv
import os.path
import glob
import PIL.Image as Image

def test_for_batch(model_name, model, test_input_dir, test_save_dir):
    model.load_weights(model_name + '.hdf5')
    data_name = os.listdir(test_input_dir)
    threshold=0.5
    for i in data_name:
        if 'predict' not in i:
            img = imageio.imread(os.path.join(test_input_dir, i), as_gray=False) 
            img = trans.resize(img, (512, 512), mode='edge')
            img = np.reshape(img, (1,) + img.shape)
            results = model.predict(img, verbose=1)
            results = np.squeeze(results)
            results[results <= threshold] = 0
            results[results > threshold] = 1
            filename = i.split('.')[0]
            imageio.imsave(os.path.join(test_save_dir, filename + ".jpg"), results)

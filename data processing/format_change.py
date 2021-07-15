import os
from PIL import Image as img
from tqdm import tqdm


path = 'binary/bw/'  # the folder with original images
path2 = 'binary/jpg/'  # the destination folder

files = os.listdir(path)
for n, filename in tqdm(enumerate(files), total=len(files)):
    # print(filename)
    png = img.open(path + filename)
    file_string = os.path.splitext(filename)[0]
    temp = file_string.split('.')  # split string at ‘.’
    png.save(path2 + temp[0] + ".jpg")  # convert to specific type images

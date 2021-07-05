# FCS-Net
Automatic defects detection of steel infrastructures in structural health monitoring (SHM) is still challenging because of complicated background, non-uniform illumination, irregular shapes and interference in images. Conventional defects detection mainly relies on manual inspection which is time-consuming and error-prone. Traditional machine-learning based approaches have good recognition performance on the simple crack image while it requires intervention and empirical judgment by experts.
# Network Architecture
![image](https://user-images.githubusercontent.com/77284145/124390189-ff7ae900-dd1c-11eb-9e7d-9963db7040a5.png)
# Data
Image annotation is an important task in deep learning models and application construction. In order to make deep learning algorithms have the ability to recognize certain objects, it is necessary to provide a large number of labeled data to train the network, and the labeled data is feed to the network to help the network to gain the ability to recognize the cracks. 200 images consist bridge steel girders with cracks provided by the IPC-SHM with the size of 4,928×3,264 or 5,152×3,864 pixels. The original full-size image were cut to smaller size to fit the model input size and flip, crop, stretch to increase enrich the training sets of the datasets. Among them, 160 images were used to establish training set, while the remaining 40 images were used to build the test set to test the accuracy of the model. The method of annotation for the image is to let the professional civil engineering technicians classify the cracks manually to ensure that there is no deviation between the crack image and the non-crack image of the datasets. the Figure demonstrates the processing of data annotation by using Labelme software.
![image](https://user-images.githubusercontent.com/77284145/124390365-d444c980-dd1d-11eb-8583-682d2c0fe6af.png)
# Enviroment
Please run pip install -r requirements.txt 
# Procedure
1.Please download the data through this link [train data](https://drive.google.com/file/d/1HCBkfUivl0bmrHkS32urTjQP2hbNhQby/view?usp=sharing), [test data](https://drive.google.com/file/d/1-hJwkERcLR6HrmVZFxq1j7vNk9V9jP6K/view?usp=sharing) and put the data at the root of the project  
If you try to use your own data, please use following format  
 >--Data  
 >> --train  
 >>> --image  
 >>> --mask  
 
 >> --test  
 >>> --image  
 >>> --mask  


2. Config the specification of the training process in train.py (e.g. epochs, steps) and run python train.py  
3. After the training process is complete, the h5 file with saved weight will generated at the root folder, please run the evaluation.py to evaluate the performance of the model.  
4. the results of cracks segmentation will generate automatically and save at the folder in /data/test/test_results/.  
# Citation

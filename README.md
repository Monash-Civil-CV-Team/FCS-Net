# FCS-Net
Automatic defects detection of steel infrastructures in structural health monitoring (SHM) is still challenging because of complicated background, non-uniform illumination, irregular shapes and interference in images. Conventional defects detection mainly relies on manual inspection which is time-consuming and error-prone. Traditional machine-learning based approaches have good recognition performance on the simple crack image while it requires intervention and empirical judgment by experts.
# Network Architecture
![image](https://user-images.githubusercontent.com/77284145/124383927-9fc31480-dd01-11eb-90c9-e2f8f9ff85d3.png)
# Enviroment
Please run pip install -r requirements.txt 
# Procedure
1.Please download the data through this link and put the data at the root of the project
If you try to use your own data, please use following format
 --Data
   --train
     --image
     --mask
   --test
     --image
     --mask
2. Config the specification of the training process in train.py (e.g. epochs, steps) and run python train.py
3. After the training process is complete, the h5 file with saved weight will generated at the root folder, please run the evaluation.py to evaluate the performance of the model.
4. the results of cracks segmentation will generate automatically and save at the folder in /data/test/test_results/.

# Flexible Multi-Task Deep Learning for Lane-Aware Panoptic Segmentation in Autonomous Driving 

## **Official Implementation for *Flexible Multi-Task Deep Learning for Lane-Aware Panoptic Segmentation in Autonomous Driving* submitted to the PeerJ Journal**
## [[LER Dataset Labels](https://drive.google.com/drive/folders/17ur_H7CkwFBAZug-QVM4NauiBSPmAl4b?usp=drive_link)] 


The LER segmentation inclusive VPS presented in the article is a novel dataset called LER dataset which is based on Drivable Area Detection (DAD) as part of the BDD100K dataset. The labels of the modified dataset can be downloaded from the Google Drive link provided in this repository. The original dataset's labels and the original RGB images shall be downloaded from the BSS100k dataset's [official website](https://bair.berkeley.edu/blog/2018/05/30/bdd).

The file "color_correction_for_LER_dataset_creation.m" is a MATLAB script to convert the BDD Drivable Area Detection maks to LER masks. The LER dataset label are already provided in the repo. This code may be modified for re-purposing to different applications.

The contents of the repo are tested under Python 3.7, PyTorch 1.12, Cuda 10.2, and mmcv==0.2.14

### Installation
Most of the codes used in the repo are based on [mmdetection](https://github.com/open-mmlab/mmdetection) commit hash 4357697. The following modifications are made in the mmdetection module.

The following commands can be used to install the dependencies after having installed Anaconda Python distribution whcih comes with modules like numpy, matplotlib:

```
conda create -n decoup_vps python=3.7 -y
conda activate decoup_vps
conda install pytorch=1.12 torchvision cudatoolkit=10.2 -c pytorch -y
pip install mmcv==0.2.14
pip install terminaltables
pip install pycocotools
pip install imagecorruptions
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install "git+https://github.com/cocodataset/panopticapi.git"
pip install -v -e .

```

The following are the steps involved to get inferences from Paddleseg for the semantic segmentation network PPLITESEG 

Ensure that the pre-requisite installations in the "requirements.txt" are made. 

```
python -m pip install paddlepaddle-gpu==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

```

After installing paddleseg the following modifications should be made in the paddle seg directory. All the required files for modification are made available in this repo.

1.The configuration for training reported in the paper are made available as yaml files. Replace those files in the path "paddleseg_root/configs/pp_liteseg" with the ones in this repo.

2. The semantic weights are basically implemented in the loss function (OhEM Cross Entropy loss) in the path "padddleseg_root/paddleseg/models/losses/ohem_cross_entropy_loss.py". Replce the original file with the one made available in this repo for implementing semantic weights for LER segmentation. Line numbers 96 to 116 may be suitably modified to either perform plain (no semantic weights) training or the semantic weights-based training. By defauly the code has semantic weights set to (1, 2, 3).
   
3. Change the dataset path to the location where the LER dataset is made available
   
4. Replace the file "logger.py" in the location "padddleseg_root/paddleseg/utils" with logger_subha.py provided in this repo/
   
5. Use the "train_subha.py" instead of "train.py" to initiate the training.
   
6. Use "evaluate.py" and "predict.py" as originally available in the paddleseg. No edits required for evaluation and prediction.
    


The details about linking regular VPS with LER shall be found in the related work presented in this [repo](https://github.com/SubhasreePasupathi/Decoupled_VPS)

The various VPS datasets used are linked below:

### [[Modified Dataset](https://drive.google.com/drive/folders/1T2hEF7VbFGRytLEMxsbDKYg-ehi8NmUg?usp=sharing)] [[Original Cityscapes VPS Dataset](https://www.dropbox.com/scl/fi/th8t12uvalox9fopzlab1/cityscapes-vps-dataset-1.0.zip?rlkey=rfd1prz6jsn4kxi1nc04gqqsr&e=1&dl=0)] [[Original KITTI STEP Dataset](https://www.cvlibs.net/datasets/kitti/eval_step.php)] [[Original Waymo PVPS Dataset](https://waymo.com/open/download)]


**EVALUATION**

All the codes related to evaluation metrics as part of the paddleseg API. The original evaluate.py file shall be used for the same.


# Flexible Multi-Task Deep Learning for Lane-Aware Panoptic Segmentation in Autonomous Driving 

## **Official Implementation for *Flexible Multi-Task Deep Learning for Lane-Aware Panoptic Segmentation in Autonomous Driving* submitted to the PeerJ Journal**
## [[LER Dataset Labels](https://drive.google.com/drive/folders/17ur_H7CkwFBAZug-QVM4NauiBSPmAl4b?usp=drive_link)] 

The repo has the code and data of the novel Left, Ego, Right (LER) Segmentation inclusive VPS.

### Preamble
The LER segmentation inclusive VPS presented in the article is a novel dataset  called LEr dataset which is based on Drivable Area Detection (DAD) as part of the BDD100K dataset. The labels of the modified dataset can be downloaded from the Google Drive link provided in this repository. The original dataset's labels and the original RGB images shall be downloaded from the BSS100k dataset's official website. (https://bair.berkeley.edu/blog/2018/05/30/bdd/)
### Disclaimer
The contents of the repo are tested under Python 3.7, PyTorch 1.12, Cuda 10.2, and mmcv==0.2.14

### Installation
Most of the codes used in the repo are based on [mmdetection](https://github.com/open-mmlab/mmdetection) commit hash 4357697. The following modifications are made in the mmdetection module.
The following commands can be used to install the dependencies for reimplementation of VPSNetafter having installed Anaconda Python distribution whcih comes with modules like numpy, matplotlib:

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

The following are the steps involved to get inferences from Paddleseg for the semantic segmentation network SSEG (a modified version of Deeplab V3+ as reported in the [paper](https://arxiv.org/pdf/1812.01593.pdf)

```
python -m pip install paddlepaddle-gpu==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

The following are the steps involved to get inferences from the official implementation for the MOTS network PCAN:

```
pip install mmcv-full==1.2.7
pip install mmdet==2.10.0
pip install motmetrics
```

Clone the github repo for [PCAN](https://github.com/SysCV/pcan/tree/main).

Run the following line to set the requirements for PCAN inferences

```
python setup.py develop
```

For the object detection YOLOv5 is used followed by BoTSort for tracking which is implemented using the Ultralytics framework. The procedure mentioned in [link](https://github.com/ultralytics/yolov5) can be followed to setup.

**Format Conversions**

The following modifications are made in the datasets to create a common basis (Cityscapes VPS convention)
1. Renumbering of class ids
2. Eliminating the extra classes other than the ones considered (listed in the journal article)
3. Reorganization of the instance mask representation to follow the Cityscapes VPS convention.


The following  changes needs to be made in order to repeat the work presented in the article published.

Use the code from the folder *format_conversion_for_inst_eval* to generate instance masks based on (1) Deepmac and class id generated from object detection matching and (2) PCAN instance masks converted to a common convention for pixel level marking of class and instance ids.

Use the code from the folder *format_conversion_for_kittti_step_vpq* to convert the KITTI STEP data to the coomon format for applying video panoptic quality evaluation metrics for .

Use the code from the folder *formats_conversion_for_track_eval* to convert all the data to the coomon format for applying evaluation metrics related to tracking performance.

Use the codes in *modify_waymoids_to_cv_ids* to convert the instance ids of waymo masks to standard format used in the paper.


**Merging Approaches (MS and MIS)**

The codes required for the mask merging of instance masks and segmentation masks along with tracking data in both MS and MIS approaches are presented in *mask_merge* folder.


**EVALUATION**

All the codes related to evaluation metrics reported in the article are presented in the folder *eval*

Codes for VPQ metrics are presented in the vpq folder which will take care of the adoption required for the KITTI STEP and Waymo datasets.


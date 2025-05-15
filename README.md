
# Flexible Multi-Task Deep Learning for Lane-Aware Panoptic Segmentation in Autonomous Driving 

## **Official Implementation for *Flexible Multi-Task Deep Learning for Lane-Aware Panoptic Segmentation in Autonomous Driving* submitted to the PeerJ Journal**
## [[LER Dataset Labels](https://drive.google.com/drive/folders/17ur_H7CkwFBAZug-QVM4NauiBSPmAl4b?usp=drive_link)] 

The repo has the code and data of the novel Left, Ego, Right (LER) Segmentation inclusive VPS.

### Preamble
The LER segmentation inclusive VPS presented in the article is a novel dataset  called LEr dataset which is based on Drivable Area Detection (DAD) as part of the BDD100K dataset. The labels of the modified dataset can be downloaded from the Google Drive link provided in this repository. The original dataset's labels and the original RGB images shall be downloaded from the BSS100k dataset's [official website](https://bair.berkeley.edu/blog/2018/05/30/bdd).
### Disclaimer
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
```
python -m pip install paddlepaddle-gpu==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
The details about linking regular VPS with LER shall be found in the related work presented in this [repo](https://github.com/SubhasreePasupathi/Decoupled_VPS)
```
**EVALUATION**

All the codes related to evaluation metrics as part of the paddleseg API.



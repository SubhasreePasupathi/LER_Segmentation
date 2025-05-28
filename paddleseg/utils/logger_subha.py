#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:24:48 2022

@author: subha
"""

import os
import time
import csv
import sys
import json
import shutil

import paddle
from pathlib import Path

#save_dir="chumma/train_ddrnet_bdd"

class Logger():
    def __init__(self, dirname,flush_n=100):
        self.root_dir="C:/Users/Admin/PaddleSeg-release-2.7/"  
        #os.chmod(self.root_dir,777)
        time_part=time.strftime("%Y%m%d-%H%M%S")
        self.log_dir=os.path.join(self.root_dir,dirname+"_log",time_part+"_log")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.json_name=os.path.join(self.log_dir,time_part+".json")
        self.csv_tr_name=os.path.join(self.log_dir,time_part+"_train.csv")
        self.csv_ev_name=os.path.join(self.log_dir,time_part+"_best_eval.csv")
        self.csv_ev_all_name=os.path.join(self.log_dir,time_part+"_eval_all.csv")
        self.log_name=os.path.join(self.log_dir,time_part+".log")
        self.csv_cols_tr=['epoch','iter','loss','lr','batch_cost','reader_cost','ips'] 
        self.csv_cols_ev=['mIoU','acc','kappa','mdice','cls_iou','cls_pr',
                                         'cls_re','best_mIoU_model_iter']
        self.csv_cols_ev_all=['iter','mIoU','acc','kappa','mdice','cls_iou','cls_pr',
                                         'cls_re']
        self.train_metrics=[]
        self.eval_metrics=[]
        self.flush_n=flush_n
        self.levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
        self.log_level = 2
        
        
        
    def write_json(self,env_info,seed,cfg):
        json_dict={'env_info': env_info,'seed':seed,'config':cfg}
        with open(self.json_name,'w')as f:
            json.dump(json_dict,f)
        
        
    
    def save_tr(self):
        print("writing")
        with open(self.csv_tr_name,'w',newline="")as f:
            writer= csv.DictWriter(f, fieldnames=self.csv_cols_tr)
            writer.writeheader()
            writer.writerows(self.train_metrics)
    
    
    
    def log_train_metrics(self,csv_dict):        
        self.train_metrics.append(csv_dict)
        if (((csv_dict['iter'])%self.flush_n)==0):
            self.save_tr()
    
    def log_eval_metrics(self,csv_dict):    
        
        print("writing_ev")
        with open(self.csv_ev_name,'w',newline="")as f:
            writer= csv.DictWriter(f, fieldnames=self.csv_cols_ev)
            writer.writeheader()
            writer.writerow(csv_dict)

    def log_eval_metrics_all(self,csv_dict):    
        
        print("writing_ev")
        with open(self.csv_ev_all_name,'a',newline="")as f:
            writer= csv.DictWriter(f, fieldnames=self.csv_cols_ev_all)
            writer.writeheader()
            writer.writerow(csv_dict)
    
    def log(self,level=2, message=""):
        if paddle.distributed.ParallelEnv().local_rank == 0:
            current_time = time.time()
            time_array = time.localtime(current_time)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
            if self.log_level >= level:
                print("{} [{}]\t{}".format(current_time, self.levels[level], message)
                      .encode("utf-8").decode("latin1"))
    def debug(self,message=""):
        self.log(level=3, message=message)
    
    
    def info(self,message=""):
        self.log(level=2, message=message)
    
    
    def warning(self,message=""):
        self.log(level=1, message=message)
    
    
    def error(self,message=""):
        self.log(level=0, message=message)
        
        
   
        
        
     
'''   
     
logg=Logger(save_dir)


env_info="sys.platform: linux\nPython: 3.8.13 (default, Mar 28 2022, 11:38:47) [GCC 7.5.0]\nCUDA available: False\nGCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0\nPyTorch: 1.10.2\nPyTorch compiling details: PyTorch built with:\n  - GCC 7.3\n  - C++ Version: 201402\n  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications\n  - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)\n  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.2, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, \n\nTorchVision: 0.11.3\nOpenCV: 4.6.0\nMMCV: 1.5.3\nMMCV Compiler: GCC 7.3\nMMCV CUDA Compiler: 11.3\nMMClassification: 0.23.1+unknown"
seed=2102588628
config="model = dict(\n    type='ImageClassifier',\n    backbone=dict(\n        type='EfficientNet',\n        arch='b7',\n        init_cfg=dict(\n            type='Pretrained',\n            checkpoint=\n            'efficientnet-b7_3rdparty_8xb32-aa-advprop_in1k_20220119-c6dbff10.pth',\n            prefix='backbone')),\n    neck=dict(type='GlobalAveragePooling'),\n    head=dict(\n        type='LinearClsHead',\n        num_classes=10,\n        in_channels=2560,\n        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),\n        topk=(1, 5)))\ndataset_type = 'CustomDataset'\nimg_norm_cfg = dict(\n    mean=[0.441, 0.4653, 0.3976], std=[0.1744, 0.1538, 0.1898], to_rgb=True)\ntrain_pipeline = [\n    dict(type='LoadImageFromFile'),\n    dict(\n        type='RandomResizedCrop',\n        size=600,\n        efficientnet_style=True,\n        interpolation='bicubic'),\n    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),\n    dict(\n        type='Normalize',\n        mean=[0.441, 0.4653, 0.3976],\n        std=[0.1744, 0.1538, 0.1898],\n        to_rgb=True),\n    dict(type='ImageToTensor', keys=['img']),\n    dict(type='ToTensor', keys=['gt_label']),\n    dict(type='Collect', keys=['img', 'gt_label'])\n]\ntest_pipeline = [\n    dict(type='LoadImageFromFile'),\n    dict(type='Resize', size=(256, -1)),\n    dict(\n        type='CenterCrop',\n        crop_size=600,\n        efficientnet_style=True,\n        interpolation='bicubic'),\n    dict(\n        type='Normalize',\n        mean=[0.441, 0.4653, 0.3976],\n        std=[0.1744, 0.1538, 0.1898],\n        to_rgb=True),\n    dict(type='ImageToTensor', keys=['img']),\n    dict(type='Collect', keys=['img'])\n]\ndata = dict(\n    samples_per_gpu=128,\n    workers_per_gpu=2,\n    train=dict(\n        type='CustomDataset',\n        data_prefix='data/tomato_plant_disease_dataset/train',\n        pipeline=[\n            dict(type='LoadImageFromFile'),\n            dict(\n                type='RandomResizedCrop',\n                size=600,\n                efficientnet_style=True,\n                interpolation='bicubic'),\n            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),\n            dict(\n                type='Normalize',\n                mean=[0.441, 0.4653, 0.3976],\n                std=[0.1744, 0.1538, 0.1898],\n                to_rgb=True),\n            dict(type='ImageToTensor', keys=['img']),\n            dict(type='ToTensor', keys=['gt_label']),\n            dict(type='Collect', keys=['img', 'gt_label'])\n        ],\n        ann_file=None,\n        classes=[\n            'Target_Spot', 'Septoria_leaf_spot',\n            'Spider_mites Two-spotted_spider_mite',\n            'Tomato_Yellow_Leaf_Curl_Virus', 'Late_blight', 'Leaf_Mold',\n            'Bacterial_spot', 'Tomato_mosaic_virus', 'Early_blight', 'healthy'\n        ]),\n    val=dict(\n        type='CustomDataset',\n        data_prefix='data/tomato_plant_disease_dataset/val',\n        ann_file=None,\n        pipeline=[\n            dict(type='LoadImageFromFile'),\n            dict(type='Resize', size=(256, -1)),\n            dict(\n                type='CenterCrop',\n                crop_size=600,\n                efficientnet_style=True,\n                interpolation='bicubic'),\n            dict(\n                type='Normalize',\n                mean=[0.441, 0.4653, 0.3976],\n                std=[0.1744, 0.1538, 0.1898],\n                to_rgb=True),\n            dict(type='ImageToTensor', keys=['img']),\n            dict(type='Collect', keys=['img'])\n        ],\n        classes=[\n            'Target_Spot', 'Septoria_leaf_spot',\n            'Spider_mites Two-spotted_spider_mite',\n            'Tomato_Yellow_Leaf_Curl_Virus', 'Late_blight', 'Leaf_Mold',\n            'Bacterial_spot', 'Tomato_mosaic_virus', 'Early_blight', 'healthy'\n        ]),\n    test=dict(\n        type='CustomDataset',\n        data_prefix='data/tomato_plant_disease_dataset/val',\n        ann_file=None,\n        pipeline=[\n            dict(type='LoadImageFromFile'),\n            dict(type='Resize', size=(256, -1)),\n            dict(\n                type='CenterCrop',\n                crop_size=600,\n                efficientnet_style=True,\n                interpolation='bicubic'),\n            dict(\n                type='Normalize',\n                mean=[0.441, 0.4653, 0.3976],\n                std=[0.1744, 0.1538, 0.1898],\n                to_rgb=True),\n            dict(type='ImageToTensor', keys=['img']),\n            dict(type='Collect', keys=['img'])\n        ],\n        classes=[\n            'Target_Spot', 'Septoria_leaf_spot',\n            'Spider_mites Two-spotted_spider_mite',\n            'Tomato_Yellow_Leaf_Curl_Virus', 'Late_blight', 'Leaf_Mold',\n            'Bacterial_spot', 'Tomato_mosaic_virus', 'Early_blight', 'healthy'\n        ]))\nevaluation = dict(interval=1, metric='accuracy')\noptimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)\noptimizer_config = dict(grad_clip=None)\nlr_config = dict(policy='step', step=[15])\nrunner = dict(type='EpochBasedRunner', max_epochs=200)\ncheckpoint_config = dict(interval=1)\nlog_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])\ndist_params = dict(backend='nccl')\nlog_level = 'INFO'\nload_from = None\nresume_from = None\nworkflow = [('train', 1)]\nclasses = [\n    'Target_Spot', 'Septoria_leaf_spot',\n    'Spider_mites Two-spotted_spider_mite', 'Tomato_Yellow_Leaf_Curl_Virus',\n    'Late_blight', 'Leaf_Mold', 'Bacterial_spot', 'Tomato_mosaic_virus',\n    'Early_blight', 'healthy'\n]\nwork_dir = './work_dirs/tomamto_efficientNet_config'\ngpu_ids = [0]\nseed = 2102588628\n"
logg.write_json(env_info,seed,config)
csv_dict={'epoch': 1, 'iter': 106, 'loss': 1.4524, 'lr': 0.009998, 'batch_cost': 15.0347, 'reader_cost': 0.03696, 'ips': 0.1995}
logg.log_train_metrics(csv_dict)


csv_dict={'epoch': 1, 'iter': 200, 'loss': 1.4524, 'lr': 0.009998, 'batch_cost': 15.0347, 'reader_cost': 0.03696, 'ips': 0.1995}
logg.log_train_metrics(csv_dict)

csv_dict={'mIoU':0.7657567,'Acc':0.98667,'best_model_iter':1677}

logg.log_eval_metrics(csv_dict)
'''

import os
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from collections import Counter
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader
import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
import training_v1
'''
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset
'''
#import training_v1
from training_v1 import Trainer, CustomImageFolder

import tqdm

device='cpu'

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM,  FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchvision.transforms.v2 as transforms
import random
import matplotlib.patches as patches

def gradcam(path, part, l_idx, img_idx=None):
    model_name = path.replace('/', '_') + '_' + part
    client = MlflowClient()
    model_version = client.search_model_versions(f'name="{model_name}"')[0].version
    
    model = mlflow.pytorch.load_model(f'models:/{model_name}/{model_version}').to('cpu') 
    model.eval()
    img_channels = Trainer.get_img_shape(f'{path}/{part}')[0]
    transform = transforms.Compose([
                                    transforms.ToImage(),
                                    transforms.ToDtype(torch.float32, scale=True),
                                    transforms.Normalize([0.5]*img_channels, [0.5]*img_channels) 
                                    ])
    
    ds = CustomImageFolder(path, transform=transform)
    ds = Subset(ds, [i for i, target in enumerate(ds.targets) if target == ds.class_to_idx[part]])
    if img_idx is None:
        img_idx = random.randint(0, len(ds.indices)-1)
    
    input_tensor = ds[img_idx][0].unsqueeze(0)
    
    img = ds[img_idx][0] * 0.5 + 0.5
    img = img.clamp(0, 1)
    
    if img_channels == 1:
        img = img.squeeze(0).numpy()  # shape (H, W)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # shape (H, W, 3)
    else:
        img = img.permute(1, 2, 0).numpy()    
    
    target_layers = [model.layers[l_idx]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)  
    #cam = HiResCAM(model=model, target_layers=target_layers)  # AblationCAM - slow
    targets = [ClassifierOutputTarget(0)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  
    avg = np.average(grayscale_cam)
    sum = np.sum(grayscale_cam[grayscale_cam > avg])
    #print(img_idx, sum, avg, np.std(grayscale_cam))
    f, ax = plt.subplots(1, 3, figsize=(10, 5)) #(40, 20)) 
    ax[2].imshow(grayscale_cam, cmap='jet') 
    try:
        top, bottom, left, right, grayscale_cam = get_boundary_box(grayscale_cam, 6)
        #print(grayscale_cam.shape, np.min(grayscale_cam), np.max(grayscale_cam))
    except ValueError:
        plt.imshow(grayscale_cam, cmap='jet')
        plt.show()
        print('Image idx:', img_idx)
        raise
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=False)
    
    #f, ax = plt.subplots(1, 3, figsize=(10, 5)) 
    ax[0].imshow(img)
    ax[1].imshow(grayscale_cam, cmap='jet')    
    #ax[2].imshow(visualization, cmap='jet')  
    
    for i in range(3):
        rect = patches.Rectangle((left, top), right-left, bottom-top, linewidth=1, edgecolor='r', facecolor='none')    
        ax[i].add_patch(rect)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
    plt.tight_layout()
    plt.show()

from scipy.ndimage import uniform_filter, median_filter, gaussian_filter

def get_boundary_box(grayscale_cam, num):    
    box = {'bottom': 0,
           'top': 0,           
           'right': 0,
           'left': 0}

    if np.sum(grayscale_cam) > 0 or True:  
        center_region = grayscale_cam[grayscale_cam.shape[0]//3:grayscale_cam.shape[0]*2//3,
                                      grayscale_cam.shape[1]//3:grayscale_cam.shape[1]*2//3]
        box['top'], box['left'] = np.unravel_index(np.argmax(center_region), center_region.shape)
        box['top'] += grayscale_cam.shape[0]//3
        box['left'] += grayscale_cam.shape[1]//3
        box['bottom'] = box['top'] + 1
        box['right'] = box['left'] + 1
        
        step = {'top': -1,
                'bottom': 1,
                'left': -1,
                'right': 1}
        
        max_vals = {}

        while True:
            for key in box:
                if key in ['top', 'bottom']:
                    if (0 <= box[key] + step[key] <= grayscale_cam.shape[0] - 1 and
                            (grayscale_cam.shape[0] // (box['bottom'] - box['top'] + 1)) * (grayscale_cam.shape[1] // (box['right'] - box['left'])) >= num): 
                        max_vals[key] = np.max(grayscale_cam[box[key] + step[key], box['left']:box['right']])
                    else:
                        max_vals[key] = -1
                else:
                    if (0 <= box[key] + step[key] <= grayscale_cam.shape[1] - 1 and
                            (grayscale_cam.shape[0] // (box['bottom'] - box['top'])) * (grayscale_cam.shape[1] // (box['right'] - box['left'] + 1)) >= num): 
                        max_vals[key] = np.max(grayscale_cam[box['top']:box['bottom'], box[key] + step[key]])
                    else:
                        max_vals[key] = -1
            
            max_key = max(max_vals, key=max_vals.get)

            if max_vals[max_key] == -1:
                break
            
            box[max_key] += step[max_key]
            
    return box['top'], box['bottom'], box['left'], box['right'], grayscale_cam

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM,  FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchvision.transforms.v2 as transforms
import random
import matplotlib.patches as patches

def gradcam_imgs(path, l_idx):
    client = MlflowClient()
    
    img_shape = Trainer.get_img_shape(f'{path}/assembly')
    transform = transforms.Compose([
                                    transforms.ToImage(),
                                    transforms.ToDtype(torch.float32, scale=True),
                                    transforms.Normalize([0.5]*img_shape[0], [0.5]*img_shape[0]) 
                                    ])
    
    ds = CustomImageFolder(path, transform=transform)
    to_pil = transforms.ToPILImage()
    dest_path = f'D:/test_img/{path.replace('./temp/', '')}'
    
    for part in ds.classes:        
        part_ds = Subset(ds, [i for i, target in enumerate(ds.targets) if target == ds.class_to_idx[part]])

        model_name = path.replace('/', '_') + '_' + part
        #print(model_name)
        model_version = client.search_model_versions(f'name="{model_name}"')[0].version        
        model = mlflow.pytorch.load_model(f'models:/{model_name}/{model_version}').to('cpu') 
        model.eval()
        
        img_aspects = []

        for i in range(len(part_ds)):
            input_tensor = part_ds[i][0].unsqueeze(0)
            
            img = part_ds[i][0] * 0.5 + 0.5
            img = img.clamp(0, 1)
            
            if img_shape[0] == 1:
                img = img.squeeze(0).numpy()  # shape (H, W)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # shape (H, W, 3)
            else:
                img = img.permute(1, 2, 0).numpy()    
            
            target_layers = [model.layers[l_idx]]
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers)  
            targets = [ClassifierOutputTarget(0)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]  
            
            top, bottom, left, right, grayscale_cam = get_boundary_box(grayscale_cam, 6)            
            pil_image = to_pil(img[top:bottom, left:right])

            img_aspect = f'{img_shape[1]//(bottom-top)}_{img_shape[2]//(right-left)}'
            img_path = f'{dest_path}/{part}/{img_aspect}'

            if not img_aspect in img_aspects:
                img_aspects.append(img_aspect)
                os.makedirs(img_path)
            
            #print(img_path)
            
            pil_image.save(f'{img_path}/{i}.png')
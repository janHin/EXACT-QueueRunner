import torch
from fastai.data.all import *
from fastai.vision.all import *
from torchvision import models
from fastai.vision.models.unet import DynamicUnet
from torchvision import transforms
import openslide
import logging
import gc
import numpy as np
from queue import Queue
from typing import Callable
import os
#from typing import Tuple, List
import cv2

def pil2tensor(image:Union[np.ndarray,np.ndarray],dtype:np.dtype)->Tensor:
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )

def custom_model_load(checkpoint, model_keys):
    updated_dict = checkpoint["model"]
    renamed_dict = {}
    for k, (key, value) in enumerate(updated_dict.items()):
        renamed_dict[model_keys[k]] = value
    return renamed_dict

def get_patch(slide, x, y, patch_size, level):
    rgb = np.array(
        slide.read_region(location=(x, y), level=level, size=(patch_size, patch_size)))
    rgb[rgb[:, :, 3] == 0] = [255, 255, 255, 0]
    rgb = rgb[:, :, :3]
    return rgb

def inference(fname, hdf5_file, update_progress:Callable, device='mps'):
    #hdf5_file = h5py.File("{}.hdf5".format(Path(fname).stem), "w")
    patch_size = 512
    down_factor = 16

    logging.info('Loading model')
    modelpath = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'catch.pth'
    backbone = models.resnet18(pretrained=True)
    body = create_body(backbone, n_in=3, pretrained=True, cut=-2)
    model = DynamicUnet(body, n_out=6, img_size=(patch_size,patch_size))
    state = custom_model_load(torch.load(modelpath, map_location=device), list(model.state_dict().keys()))
    model.load_state_dict(state)
    model = model.eval().to(device)
    
    slideobjs=[]
    
    # Preprocess WSI
    slide = openslide.open_slide(fname)
    notWSI=False
    downsamples_int = [int(x) for x in slide.level_downsamples]
    if 32 in downsamples_int:
        ds = 32
    elif 16 in downsamples_int:
        ds = 16
    else:
        ds=16
        notWSI=True
        # if it is not a WSI, all tiles are calculated

    if not notWSI:
        overview_level = np.where(np.abs(np.array(slide.level_downsamples)-ds)<0.1)[0][0]
        overview = slide.read_region(level=overview_level, location=(0,0), size=slide.level_dimensions[overview_level])

        # Convert to grayscale
        gray = cv2.cvtColor(np.array(overview)[:,:,0:3],cv2.COLOR_BGR2GRAY)

        # OTSU thresholding
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # dilate
        dil = cv2.dilate(thresh, kernel = np.ones((7,7),np.uint8))

        # erode
        activeMap = cv2.erode(dil, kernel = np.ones((7,7),np.uint8))
    else:
        # if it is not a WSI, all tiles are calculated
        activeMap = np.ones((int(slide.dimensions[1]/ds),int(slide.dimensions[0]/ds)))
        overview=np.ones((int(slide.dimensions[1]/ds),int(slide.dimensions[0]/ds),3))

    overlap=0.5
    uid=100000
    annos_original=[]

    level = (np.abs(np.array(slide.level_downsamples) - down_factor)).argmin()
    if "openslide.bounds-width" in list(slide.properties.keys()):
        new_dimensions = ((int(slide.properties["openslide.bounds-width"]) + int(slide.properties["openslide.bounds-x"])),
                        (int(slide.properties["openslide.bounds-height"]) + int(slide.properties["openslide.bounds-y"])))
    else: new_dimensions = slide.dimensions
    shape = [int(nd/slide.level_downsamples[level]) for nd in new_dimensions]
    start, stop = int(patch_size * (overlap / 2)), int(patch_size * (3 * overlap / 2))
    segmentation_results = hdf5_file.create_dataset("segmentation", (shape[1], shape[0]), dtype='uint8', compression="gzip")
    x_steps = range(0, int(slide.level_dimensions[0][0]),
                    int(patch_size * down_factor * overlap))
    y_steps = range(0, int(slide.level_dimensions[0][1]),
                    int(patch_size * down_factor * overlap))
    patches = []
    x_coordinates = []
    y_coordinates = []
    batch_size=4
    patch_counter = 0
    seg_pred_batch = []
    overlay = np.zeros(np.array(overview).shape, np.uint8)[:,:,0:3]
    overlay[:,:,0] = activeMap
    mean = torch.FloatTensor([0.7587, 0.5718, 0.6572])
    std = torch.FloatTensor([0.0866, 0.1118, 0.0990])
    requestQueue = Queue()


    step_ds = int(np.ceil(float(patch_size * down_factor)/ds))
    for y in y_steps:
        for x in x_steps:
            requestQueue.put((x,y,ds))

    with torch.no_grad():
        for i,y in enumerate(y_steps):
            update_progress(i/len(y_steps)*80)
            for x in x_steps:
                [x,y,ds] = requestQueue.get()
                x_ds = int(np.floor(float(x)/ds))
                y_ds = int(np.floor(float(y)/ds))
                needCalculation = np.sum(activeMap[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds])>0.9*step_ds*step_ds

                if not needCalculation:
                    patch_counter += 1
                    continue
                
                patch = get_patch(slide, int(x), int(y), patch_size, level)
                patch = pil2tensor(patch / 255., np.float32)
                patch = transforms.Normalize(mean, std)(patch)
                
                patches.append(patch[None, :, :, :])
                x_coordinates.append(x//down_factor)
                y_coordinates.append(y//down_factor)

                if len(patches) == batch_size:
                    predictions = model(torch.cat(patches).to(device))
                    seg_pred = torch.softmax(predictions, dim=1)
                    seg_pred_batch.extend (seg_pred.cpu())
                    patches = []

                patch_counter += 1
                
                
        if len(patches) > 0:
            predictions =  model(torch.cat(patches).to(device))
            seg_pred = torch.softmax(predictions, dim=1)
            seg_pred_batch.extend (seg_pred.cpu())

        counter = 0
        
        
        for idx, (pred, x, y) in enumerate(zip(seg_pred_batch, x_coordinates, y_coordinates)):
            
            if (idx%100 == 0):
                update_progress(counter/len(y_coordinates)*10+80)
            
            height, width = segmentation_results[int(y + start):int(y + stop),int(x + start):int(x + stop)].shape
            probas, labels = pred.max(dim=0)
            segmentation_results[int(y + start):int(y + stop),int(x + start):int(x + stop)] = labels[start:stop, start:stop][:height, :width].cpu()
            counter += 1
   
    # free up memory
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    update_progress(80)

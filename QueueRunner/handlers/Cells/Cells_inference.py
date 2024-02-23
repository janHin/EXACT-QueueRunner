import torch
import logging
import gc
import numpy as np
import time
from queue import Queue
from typing import Callable
import os
import torchvision.transforms as transforms

from typing import Tuple, List

from fastai.vision import *
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from fastai.vision import models
#from .object_detection_fastai.helper.object_detection_helper import *
from .object_detection_fastai.models.RetinaNet import RetinaNet
from .object_detection_fastai.helper.object_detection_helper import *

import openslide
import cv2

from torch.utils.data import Dataset, DataLoader
from typing import Union

from torchvision import transforms, utils
from torch import Tensor
import numpy as np
import torch
def pil2tensor(image:Union[np.ndarray,np.ndarray],dtype:np.dtype)->Tensor:
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class InferenceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, slide, level, coordlist, mean = torch.FloatTensor([0.7404, 0.7662, 0.7805]), std = torch.FloatTensor([0.1504, 0.1313, 0.1201])):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coordlist = coordlist
        self.slide = slide
        self.level = level
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.coordlist)

    def __getitem__(self, idx):
        shape=256
        x,y = self.coordlist[idx]
        patch = np.array(self.slide.read_region(location=(int(x), int(y)),
                                                level=self.level, size=(shape, shape)))[:, :, :3]


        patch = pil2tensor(patch / 255., np.float32)
        patch = transforms.Normalize(self.mean, self.std)(patch)
        

        return patch, x, y    
    

def inference(fname, update_progress:Callable, stage1_threshold:float=0.64, nms_thresh=0.5, device='mps'):


    logging.info('Loading model')
    modelpath = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'pan_tumor.pth'
    anchors = create_anchors(sizes=[(32, 32), (16, 16), (8, 8), (4, 4)], ratios=[0.5, 1, 2],scales=[0.5, 0.75, 1, 1.25, 1.5])
    encoder = create_body(resnet18(), pretrained=False, cut=-2)
    # Careful: Number of anchors has to be adapted to scales
    model = RetinaNet(encoder, n_classes=4, n_anchors=15, sizes=[32, 16, 8, 4], chs=128, final_bias=-4., n_conv=3)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu'))['model'])
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
        level = np.where(np.abs(np.array(slide.level_downsamples)-ds)<0.1)[0][0]
        overview = slide.read_region(level=level, location=(0,0), size=slide.level_dimensions[level])

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

    down_factor=1
    overlap=0.9
    shape = 256
    uid=100000
    annos_original=[]
    level=0
    # TODO change interval
    #x_steps = range(0, int(slide.level_dimensions[0][0]),
    #                int(shape * down_factor * overlap))
    #y_steps = range(0, int(slide.level_dimensions[0][1]),
    #                int(shape * down_factor * overlap))

    xmin = int(slide.level_dimensions[0][0]//2)
    ymin = int(slide.level_dimensions[0][1]//2)
    x_steps = range(xmin, xmin+5000,
                int(shape * down_factor * overlap))
    y_steps = range(ymin, ymin+5000,
                    int(shape * down_factor * overlap))
    
    patches = []
    x_coordinates = []
    y_coordinates = []
    batch_size=4
    patch_counter = 0
    class_pred_batch, bbox_pred_batch = [], []
    overlay = np.zeros(np.array(overview).shape, np.uint8)[:,:,0:3]
    overlay[:,:,0] = activeMap
    
    requestQueue = Queue()

    coordlist = []
    step_ds = int(np.ceil(float(shape)/ds))
    for y in y_steps:
        for x in x_steps:
            x_ds = int(np.floor(float(x)/ds))
            y_ds = int(np.floor(float(y)/ds))
            step_ds = int(np.ceil(float(shape)/ds))
            needCalculation = np.sum(activeMap[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds])>0.9*step_ds*step_ds
            if (needCalculation):
                coordlist.append([x,y])


    logging.info('Running inference on:'+str(device))
    logging.info('Maximum number of patches is: %d x %d = %d' % (len(x_steps), len(y_steps), len(x_steps)*len(y_steps)))

    ds = InferenceDataset(slide,level=0,coordlist=coordlist)
    
    time_reading=0
    time_processing=0
    dl = DataLoader(ds, num_workers=0, batch_size=8)
    i=0
    t0=time.time()
    with torch.inference_mode():
        for patches,x,y in dl:
            time_reading += time.time()-t0
            i+=len(patches)
            if (i%5==0): # too many API calls slow the system down
                update_progress(i/len(coordlist)*80)
            if (i%10==0):
                logging.info('Time for reading: %.2f seconds, time for processing: %.2f seconds' % (time_reading,time_processing))
             
            t0=time.time()
            mdlout = model(patches.to(device))
            class_pred, bbox_pred, _ = mdlout
            class_pred_batch.extend (class_pred.cpu())
            bbox_pred_batch.extend(bbox_pred.cpu())
            x_coordinates.extend(x)
            y_coordinates.extend(y)
            time_processing += time.time()-t0
            t0=time.time()
            patch_counter += 1


        logging.info(f'Ran inference for {patch_counter} patches.')

        counter = 0        
        for idx, (clas_pred, bbox_pred, x, y) in enumerate(zip(class_pred_batch, bbox_pred_batch, x_coordinates, y_coordinates)):
            
            if (idx%100 == 0):
                update_progress(counter/len(y_coordinates)*10+80)


            modelOutput = process_output(clas_pred.cpu(), bbox_pred.cpu(),
                                                    anchors, detect_thresh=0.4)

            bbox_pred, scores, preds = modelOutput
            if bbox_pred is not None:
                to_keep = nms(bbox_pred, scores, nms_thresh)  # nms_thresh=
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()

                t_sz = torch.Tensor([shape, shape])[None].float()
                bbox_pred = rescale_box(bbox_pred, t_sz)

                for box, pred, score in zip(bbox_pred, preds, scores):
                    y_box, x_box = box[:2]
                    h, w = box[2:4]

                    x1 = int(x_box) * down_factor + x
                    y1 = int(y_box) * down_factor + y
                    x2 = x1 + int(w) * down_factor
                    y2 = y1 + int(h) * down_factor

                    annos_original.append([x1, y1, x2, y2, float(score), int(pred)])

                    uid += 1

            counter += 1
   
    # free up memory
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    update_progress(80)
    return annos_original

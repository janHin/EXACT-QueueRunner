from torch.utils.data import Dataset, DataLoader
from ..utils.object_detection_helper import *
from typing import Callable, Union
from torchvision import transforms
from pathlib import Path
import numpy as np
import openslide
import logging
import pyvips
import torch
import time
import cv2
import gc
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def pil2tensor(image:Union[np.ndarray,np.ndarray],dtype:np.dtype)->torch.Tensor:
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )

class InferenceDataset(Dataset):
    def __init__(self, slide, level, patch_size, coordlist, mean, std):
        self.coordlist = coordlist
        self.slide = slide
        self.level = level
        self.patch_size = patch_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.coordlist)

    def __getitem__(self, idx):
        x,y = self.coordlist[idx]
        patch = np.array(self.slide.read_region(location=(int(x), int(y)),
                                                level=self.level, size=(self.patch_size, self.patch_size)))[:, :, :3]
        patch = pil2tensor(patch / 255., np.float32)
        if self.mean is not None and self.std is not None:
            patch = transforms.Normalize(self.mean, self.std)(patch)
        return patch, x, y  
    
class Inference:
    def __init__(self, fname: str, down_factor: int, patch_size: int, update_progress: Callable, mean: torch.FloatTensor = None,  std: torch.FloatTensor = None):
        self.slide = openslide.open_slide(fname)
        self.down_factor = down_factor
        self.level =  np.where(np.abs(np.array(self.slide.level_downsamples)-self.down_factor) < 1)[0][0]
        self.patch_size = patch_size
        self.device = torch.device('mps') if torch.backends.mps.is_available else torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.outputs = 1
        self.update_progress = update_progress
        self.mean = mean
        self.std = std

    def process(self):
        self.model = self.configure_model()
        activeMap = self.get_activeMap()
        coordlist = self.get_coordlist(activeMap)
        self.dataset = InferenceDataset(self.slide, level=self.level, patch_size=self.patch_size, coordlist=coordlist, mean=self.mean, std=self.std)
        x_coordinates, y_coordinates, predictions = self.inference()
        predictions = self.postprocess(predictions, x_coordinates, y_coordinates)
        annotations = self.get_annotations(predictions)

        # free up memory
        self.model.cpu()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.update_progress(95)
        return annotations

    def configure_model(self):
        raise NotImplementedError
    
    
    def get_activeMap(self):
        # Preprocess WSI
        downsamples_int = [int(x) for x in self.slide.level_downsamples]
        self.ds_map = 32 if 32 in downsamples_int else 16
        notWSI = len(downsamples_int) <= 1
        if not notWSI:
            level = np.where(np.abs(np.array(self.slide.level_downsamples)-self.ds_map)<1)[0][0]
            overview = self.slide.read_region(level=level, location=(0,0), size=self.slide.level_dimensions[level])

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
            activeMap = np.ones((int(self.slide.dimensions[1]/self.ds_map),int(self.slide.dimensions[0]/self.ds_map)))
        return activeMap
    
    def get_coordlist(self, activeMap, overlap=0.2):
        coordlist = []
        self.overlap = overlap
        step_ds = int(np.ceil(float(self.patch_size)/self.ds_map))
        x_steps = range(0, int(self.slide.level_dimensions[0][0]),
                        int(self.patch_size * self.down_factor * (1 - overlap)))
        y_steps = range(0, int(self.slide.level_dimensions[0][1]),
                        int(self.patch_size * self.down_factor * (1 - overlap)))
        for y in y_steps:
            for x in x_steps:
                x_ds = int(np.floor(float(x)/self.ds_map))
                y_ds = int(np.floor(float(y)/self.ds_map))
                step_ds = int(np.ceil(float(self.patch_size)/self.ds_map))
                needCalculation = np.sum(activeMap[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds])>0.5*step_ds*step_ds
                if (needCalculation):
                    coordlist.append([x,y])
        logging.info('Running inference on:'+str(self.device))
        logging.info('Total number of patches is %d' % (len(coordlist)))
        return coordlist
    
    def inference(self):
        time_reading=0
        time_processing=0
        t0 = time.time()
        dl = DataLoader(self.dataset, num_workers=4, batch_size=8)
        patch_counter = 0
        prediction = []
        x_coordinates, y_coordinates = [], []
        with torch.inference_mode():
            self.model.eval().to(self.device)
            for patches,x,y in dl:
                time_reading += time.time()-t0
                patch_counter += len(patches)
                if (patch_counter%5==0): # too many API calls slow the system down
                    self.update_progress(patch_counter/len(self.dataset)*80)
                if (patch_counter%10==0):
                    logging.info('Time for reading: %.2f seconds, time for processing: %.2f seconds' % (time_reading,time_processing))
                t0=time.time()
                mdlout = self.model(patches.to(self.device))
                if type(mdlout) is tuple:
                    prediction.extend(zip(*mdlout[:self.outputs]))
                else:    
                    prediction.extend(mdlout)
                x_coordinates.extend(x)
                y_coordinates.extend(y)
                time_processing += time.time()-t0
                t0=time.time()
        logging.info(f'Ran inference for {patch_counter} patches.')
        return x_coordinates, y_coordinates, prediction
    
    def postprocess(self, predictions, x_coordinates, y_coordinates):
        return predictions
    
    def get_annotations(self, predictions):
        raise NotImplementedError

class DetectionInference(Inference):
    def __init__(self, detection_threshold: float, nms_threshold: float=0.4,  **kwargs):
        super().__init__(**kwargs)
        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.outputs = 2

    def postprocess(self, predictions, x_coordinates, y_coordinates):
        counter = 0
        t_sz = torch.Tensor([self.patch_size, self.patch_size])[None].float()
        outputs = []
        for idx, ((clas_pred, bbox_pred), x, y) in enumerate(zip(predictions, x_coordinates, y_coordinates)):
            if (idx%100 == 0):
                self.update_progress(counter/len(predictions)*10+80)

            modelOutput = process_output(clas_pred.cpu(), bbox_pred.cpu(),
                                                    self.anchors, detect_thresh=self.detection_threshold)
            bbox_pred, scores, preds = modelOutput
            if bbox_pred is not None:
                to_keep = nms(bbox_pred, scores, self.nms_threshold)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()
                bbox_pred = rescale_box(bbox_pred, t_sz)
                outputs.append((bbox_pred, scores, preds, x, y))
            counter += 1
        return outputs
    
    def get_annotations(self, predictions):
        annos = []
        for idx, (bbox_pred, scores, preds, x, y) in enumerate(predictions):
            if (idx%100 == 0):
                self.update_progress(idx/len(predictions)*10+90)
            for box, score, pred in zip(bbox_pred, scores, preds):
                y_box, x_box = box[:2]
                h, w = box[2:4]

                x1 = int(x_box) * self.down_factor + x
                y1 = int(y_box) * self.down_factor + y
                x2 = x1 + int(w) * self.down_factor
                y2 = y1 + int(h) * self.down_factor

                annos.append([x1, y1, x2, y2, float(score), int(pred)])
        return annos
    

class SegmentationInference(Inference):
    def __init__(self, hdf5_file, **kwargs):
        super().__init__(**kwargs)
        self.outputs = 1
        self.hdf5_file = hdf5_file

    def get_activeMap(self):
        # Perform inference on all patches
        downsamples_int = [int(x) for x in self.slide.level_downsamples]
        self.ds_map = 32 if 32 in downsamples_int else 16
        activeMap = np.ones((int(self.slide.dimensions[1]/self.ds_map),int(self.slide.dimensions[0]/self.ds_map)))
        return activeMap

    def postprocess(self, predictions, x_coordinates, y_coordinates):
        if "openslide.bounds-width" in list(self.slide.properties.keys()):
            new_dimensions = ((int(self.slide.properties["openslide.bounds-width"]) + int(self.slide.properties["openslide.bounds-x"])),
                            (int(self.slide.properties["openslide.bounds-height"]) + int(self.slide.properties["openslide.bounds-y"])))
        else: new_dimensions = self.slide.dimensions
        shape = [int(nd/self.slide.level_downsamples[self.level]) for nd in new_dimensions]
        segmentation_results = self.hdf5_file.create_dataset("segmentation", (shape[1], shape[0]), dtype='uint8', compression="gzip")
        counter = 0
        for idx, (pred, x, y) in enumerate(zip(predictions, x_coordinates, y_coordinates)):
            if (idx%100 == 0):
                self.update_progress(counter/len(predictions)*10+80)
            x_ds, y_ds = x//self.down_factor, y//self.down_factor
            height, width = segmentation_results[int(y_ds):int(y_ds + self.patch_size),int(x_ds):int(x_ds + self.patch_size)].shape
            probas, labels = pred.max(dim=0)
            segmentation_results[int(y_ds):int(y_ds + height),int(x_ds):int(x_ds + width)] = labels[:height, :width].cpu()
            counter += 1
        return None
    
    def get_annotations(self, predictions):
        outputs = []
        for n, key in enumerate(list(self.hdf5_file.keys())):
            data = self.hdf5_file[key]
            ndarray_data = np.array(data)
            scaled_image_data = (ndarray_data * (255 / len(np.unique(ndarray_data)))).astype(np.uint8)


            # Define a color mapping for each integer value
            color_map = {
                0: (0, 0, 0, 255),    # Black
                1: (255, 0, 0, 255),  # Red
                2: (0, 255, 0, 255),  # Green
                3: (0, 0, 255, 255),  # Blue
                4: (255, 255, 0, 255),  # Yellow
                5: (255, 255, 255, 255)  # White
            }

            #colored_image = cv2.applyColorMap(ndarray_data.astype(np.uint8), colormap=color_map)
            colored_image = cv2.applyColorMap(scaled_image_data, cv2.COLORMAP_VIRIDIS)
            vi = pyvips.Image.new_from_array(colored_image)
            mask_path = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', "{}_{}.tiff".format(Path(self.slide._filename).stem, key))
            outputs.append(mask_path)
            vi.tiffsave(str(mask_path), tile=True, compression='lzw', bigtiff=True, pyramid=True, tile_width=256, tile_height=256)
        
        if self.hdf5_file.__bool__():
            self.hdf5_file.close()
        return outputs
    

import torch
import logging
import gc
import numpy as np
from queue import Queue
from typing import Callable
import os
import torchvision.transforms as transforms

from typing import Tuple, List

from .Mitosis.lib.helper.fastai_helpers import *
from .Mitosis.lib.helper.object_detection_helper import *
from .Mitosis.lib.object_detection_helper import *
from .Mitosis.lib.helper.nms import non_max_suppression_by_distance
from torchvision.models.resnet import resnet18

from fastai.vision.learner import create_body
from fastai.vision import models
from fastai.vision import *
from fastai.callback.hook import Hooks, hook_outputs

import openslide
import cv2

def rescale_box(bboxes, size: Tensor):
    bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2
    bboxes[:, :2] = (bboxes[:, :2] + 1) * size / 2
    bboxes[:, 2:] = bboxes[:, 2:] * size / 2
    bboxes = bboxes.long()
    return bboxes


# Gradient Reverse Layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    

def _get_sfs_idxs(sizes:List) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes"
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs

class LateralUpsampleMerge(nn.Module):

    def __init__(self, ch, ch_lat, hook):
        super().__init__()
        self.hook = hook
        self.conv_lat = conv2d(ch_lat, ch, ks=1, bias=True)

    def forward(self, x):
        return self.conv_lat(self.hook.stored) + F.interpolate(x, scale_factor=2)
    
class Discriminator(nn.Module):
    def __init__(self, size, n_domains, alpha=1.0):
        super(Discriminator, self).__init__()
        self.alpha = alpha
        self.reducer = nn.Sequential(
            nn.Conv2d(size, size, kernel_size = (3, 3), bias=False),
            nn.BatchNorm2d(size),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(size, size//2, kernel_size = (3, 3), bias=False),
            nn.BatchNorm2d(size//2),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(size//2, size//4, kernel_size = (3, 3), bias=False),
            nn.BatchNorm2d(size//4),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.reducer2 = nn.Linear(size//4, n_domains, bias = False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = GradReverse.apply(x, self.alpha)
        x = self.reducer(x)
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        return x


class RetinaNet(nn.Module):
    "Implements RetinaNet from https://arxiv.org/abs/1708.02002"

    def __init__(self, encoder: nn.Module, n_classes, n_domains, final_bias:float=0.,  n_conv:float=4,
                 chs=256, n_anchors=9, flatten=True, sizes=None, imsize = (256, 256)):
        super().__init__()
        self.n_classes, self.flatten = n_classes, flatten
        self.sizes = sizes
        sfs_szs, x, hooks = self._model_sizes(encoder, size=imsize)
        sfs_idxs = _get_sfs_idxs(sfs_szs)
        self.encoder = encoder
        self.c5top5 = conv2d(sfs_szs[-1][1], chs, ks=1, bias=True)
        self.c5top6 = conv2d(sfs_szs[-1][1], chs, stride=2, bias=True)
        self.p6top7 = nn.Sequential(nn.ReLU(), conv2d(chs, chs, stride=2, bias=True))
        self.merges = nn.ModuleList([LateralUpsampleMerge(chs, szs[1], hook)
                                     for szs, hook in zip(sfs_szs[-2:-4:-1], hooks[-2:-4:-1])])
        self.n_domains = n_domains
        self.smoothers = nn.ModuleList([conv2d(chs, chs, 3, bias=True) for _ in range(3)])
        self.discriminator = Discriminator(sfs_szs[-1][1], n_domains)
        self.classifier = self._head_subnet(n_classes, n_anchors, final_bias, chs=chs, n_conv=n_conv)
        self.box_regressor = self._head_subnet(4, n_anchors, 0., chs=chs, n_conv=n_conv)

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256):
        layers = [self._conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)

    def _apply_transpose(self, func, p_states, n_classes):
        if not self.flatten:
            sizes = [[p.size(0), p.size(2), p.size(3)] for p in p_states]
            return [func(p).permute(0, 2, 3, 1).view(*sz, -1, n_classes) for p, sz in zip(p_states, sizes)]
        else:
            return torch.cat(
                [func(p).permute(0, 2, 3, 1).contiguous().view(p.size(0), -1, n_classes) for p in p_states], 1)

    def _model_sizes(self, m: nn.Module, size:tuple=(256,256), full:bool=True) -> Tuple[List,Tensor,Hooks]:
        "Passes a dummy input through the model to get the various sizes"
        hooks = hook_outputs(m)
        ch_in = in_channels(m)
        x = torch.zeros(1,ch_in,*size)
        x = m.eval()(x)
        res = [o.stored.shape for o in hooks]
        if not full: hooks.remove()
        return res,x,hooks if full else res

    def _conv2d_relu(self, ni:int, nf:int, ks:int=3, stride:int=1,
                    padding:int=None, bn:bool=False, bias=True) -> nn.Sequential:
        "Create a `conv2d` layer with `nn.ReLU` activation and optional(`bn`) `nn.BatchNorm2d`"
        layers = [conv2d(ni, nf, ks=ks, stride=stride, padding=padding, bias=bias), nn.ReLU()]
        if bn: layers.append(nn.BatchNorm2d(nf))
        return nn.Sequential(*layers)

    def forward(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        for merge in self.merges:
            p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        if self.sizes is not None:
            p_states = [p_state for p_state in p_states if p_state.size()[-1] in self.sizes]
        domain = self.discriminator(c5)
        return [self._apply_transpose(self.classifier, p_states, self.n_classes),
                self._apply_transpose(self.box_regressor, p_states, 4),
                domain,
                [[p.size(2), p.size(3)] for p in p_states]]
    

def inference(fname, update_progress:Callable, stage1_threshold:float=0.55, nms_thresh=0.4, device='cuda:0'):


    logging.info('Loading model')
    
    modelpath = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'Mitosis/model.pth'
    
    scales = [0.2, 0.4, 0.6, 0.8, 1.0]
    ratios = [1]
    sizes = [(64, 64), (32, 32), (16, 16)]

    encoder = create_body(resnet18(), pretrained=False, cut=-2)
    scales = [0.2, 0.4, 0.6, 0.8, 1.0]
    ratios = [1]
    sizes = [(32, 32)]
    model = RetinaNet(encoder, n_classes=2, n_domains=6, n_anchors=len(scales) * len(ratios),sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3, imsize=(512,512))
    
    model.cpu()
    model.load_state_dict(torch.load(modelpath))
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
    shape = 512
    uid=100000
    annos_original=[]
    level=0
    x_steps = range(0, int(slide.level_dimensions[0][0]),
                    int(shape * down_factor * overlap))
    y_steps = range(0, int(slide.level_dimensions[0][1]),
                    int(shape * down_factor * overlap))
    patches = []
    x_coordinates = []
    y_coordinates = []
    batch_size=4
    patch_counter = 0
    class_pred_batch, bbox_pred_batch = [], []
    overlay = np.zeros(np.array(overview).shape, np.uint8)[:,:,0:3]
    overlay[:,:,0] = activeMap
    mean = torch.FloatTensor([0.7059, 0.4517, 0.7129])
    std = torch.FloatTensor([0.0605, 0.0852, 0.0473])
    
    requestQueue = Queue()


    step_ds = int(np.ceil(float(shape)/ds))
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
                step_ds = int(np.ceil(float(shape)/ds))
                needCalculation = np.sum(activeMap[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds])>0.9*step_ds*step_ds

                if not needCalculation:
                    patch_counter += 1
                    continue

                
                patch = np.array(slide.read_region(location=(int(x), int(y)),
                                                        level=level, size=(shape, shape)))[:, :, :3]

                patch = pil2tensor(patch / 255., np.float32)
                patch = transforms.Normalize(mean, std)(patch)
                
                
                patches.append(patch[None, :, :, :])
                x_coordinates.append(x)
                y_coordinates.append(y)

                if len(patches) == batch_size:
                    mdlout = model(torch.cat(patches).to(device))
                    class_pred, bbox_pred, domain, _ = mdlout
                    class_pred_batch.extend (class_pred.cpu())
                    bbox_pred_batch.extend(bbox_pred.cpu())

                    patches = []

                patch_counter += 1
                
                
        if len(patches) > 0:
            class_pred, bbox_pred, domain, _ = model(torch.cat(patches).to(device))
            class_pred_batch.extend (class_pred.cpu())
            bbox_pred_batch.extend(bbox_pred.cpu())


        counter = 0
        
        anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
        
        for idx, (clas_pred, bbox_pred, x, y) in enumerate(zip(class_pred_batch, bbox_pred_batch, x_coordinates, y_coordinates)):
            
            if (idx%100 == 0):
                update_progress(counter/len(y_coordinates)*10+80)


            modelOutput = process_output(clas_pred.cpu(), bbox_pred.cpu(),
                                                    anchors, detect_thresh=0.4)

            bbox_pred, scores, preds = [modelOutput[x] for x in ['bbox_pred', 'scores', 'preds']]
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

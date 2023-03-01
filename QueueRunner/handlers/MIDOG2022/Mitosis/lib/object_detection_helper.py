# Taken from C. Marzahl's repository at: 

# https://github.com/ChristianMarzahl/ObjectDetection

# Modified for fastai v2

import numpy as np
from fastai import *
from fastai.vision import *
from torch import Tensor, FloatTensor, LongTensor
import math
import torch
from typing import Any

def range_of(x):
    "Create a range from 0 to `len(x)`."
    return list(range(len(x)))


def is_tuple(x:Any)->bool: return isinstance(x, tuple)


def create_anchors(sizes, ratios, scales, flatten=True):
    "Create anchor of `sizes`, `ratios` and `scales`."
    aspects = [[[s*math.sqrt(r), s*math.sqrt(1/r)] for s in scales] for r in ratios]
    aspects = torch.tensor(aspects).view(-1,2)
    anchors = []
    for h,w in sizes:
        #4 here to have the anchors overlap.
        sized_aspects = 4 * (aspects * torch.tensor([2/h,2/w])).unsqueeze(0)
        base_grid = create_grid((h,w)).unsqueeze(1)
        n,a = base_grid.size(0),aspects.size(0)
        ancs = torch.cat([base_grid.expand(n,a,2), sized_aspects.expand(n,a,2)], 2)
        anchors.append(ancs.view(h,w,a,4))
    return torch.cat([anc.view(-1,4) for anc in anchors],0) if flatten else anchors

def create_circular_anchors(sizes, scales, flatten=True):
    "Create circular anchor grid of `sizes`, and `scales`."
    aspects = [[s] for s in scales] 
    aspects = torch.tensor(aspects)
    anchors = []
    for h,w in sizes:
        #4 here to have the anchors overlap.
        sized_aspects = 4 * (aspects * torch.tensor([2/h])).unsqueeze(0)
        base_grid = create_grid((h,w)).unsqueeze(1)
        n,a = base_grid.size(0),aspects.size(0)
        ancs = torch.cat([base_grid.expand(n,a,2), sized_aspects.expand(n,a,1)], 2)
        anchors.append(ancs.view(h,w,a,3))
    return torch.cat([anc.view(-1,3) for anc in anchors],0) if flatten else anchors


def create_grid(size):
    "Create a grid of a given `size`."
    H, W = size if is_tuple(size) else (size,size)
    grid = FloatTensor(H, W, 2)
    linear_points = torch.linspace(-1+1/W, 1-1/W, W) if W > 1 else tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(-1+1/H, 1-1/H, H) if H > 1 else tensor([0.])
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1,2)


def tlbr2cthw(boxes):
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[:,:2] + boxes[:,2:])/2
    sizes = boxes[:,2:] - boxes[:,:2]
    return torch.cat([center, sizes], 1)


def encode_class(idxs, n_classes):
    target = idxs.new_zeros(len(idxs), n_classes).float()
    mask = idxs != 0
    i1s = LongTensor(list(range(len(idxs))))
    target[i1s[mask],idxs[mask]-1] = 1
    return target




def show_anchors(ancs, size):
    _,ax = plt.subplots(1,1, figsize=(5,5))
    ax.set_xticks(np.linspace(-1,1, size[1]+1))
    ax.set_yticks(np.linspace(-1,1, size[0]+1))
    ax.grid()
    ax.scatter(ancs[:,1], ancs[:,0]) #y is first
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim(-1,1)
    ax.set_ylim(1,-1) #-1 is top, 1 is bottom
    for i, (x, y) in enumerate(zip(ancs[:, 1], ancs[:, 0])):
        ax.annotate(i, xy = (x,y))


def show_boxes(boxes):
    "Show the `boxes` (size by 4)"
    _, ax = plt.subplots(1,1, figsize=(5,5))
    ax.set_xlim(-1,1)
    ax.set_ylim(1,-1)
    for i, bbox in enumerate(boxes):
        bb = bbox.numpy()
        rect = [bb[1]-bb[3]/2, bb[0]-bb[2]/2, bb[3], bb[2]]
        draw_rect(ax, rect)


def bbox_to_activ(bboxes, anchors, flatten=True):

    "Return the target of the model on `anchors` for the `bboxes`."
    if flatten:
        # x and y offsets are normalized by radius
        t_centers = (bboxes[...,:2] - anchors[...,:2]) / anchors[...,2:]
        # Radius is given in log scale, relative to anchor radius
        t_sizes = torch.log(bboxes[...,2:] / anchors[...,2:] + 1e-8)
        # Finally, everything is divided by 0.1 (radii by 0.2)
        if (bboxes.shape[-1] == 4):
            return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
        else:
            return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2]]))

            
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res



def activ_to_bbox(acts, anchors, flatten=True):
    "Extrapolate bounding boxes on anchors from the model activations."

    if flatten:

        if (anchors.shape[-1]==4):
            acts = acts * acts.new_tensor([[0.1, 0.1, 0.2, 0.2]])
            centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
            sizes = anchors[...,2:] * torch.exp(acts[...,2:])

        else:
            acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2]]))
            centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
            sizes = anchors[...,2:] * torch.exp(acts[...,2:])
        return torch.cat([centers, sizes], -1)
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res


def cthw2tlbr(boxes):
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[:,:2] - boxes[:,2:]/2
    bot_right = boxes[:,:2] + boxes[:,2:]/2
    return torch.cat([top_left, bot_right], 1)


def intersection(anchors, targets):
    "Compute the sizes of the intersections of `anchors` by `targets`."
    ancs, tgts = cthw2tlbr(anchors), cthw2tlbr(targets)
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(a,t,4), tgts.unsqueeze(0).expand(a,t,4)
    top_left_i = torch.max(ancs[...,:2], tgts[...,:2])
    bot_right_i = torch.min(ancs[...,2:], tgts[...,2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[...,0] * sizes[...,1]
   

        


def IoU_values(anchors, targets):
    "Compute the IoU values of `anchors` by `targets`."
    if (anchors.shape[-1]==4):

        inter = intersection(anchors, targets)
        anc_sz, tgt_sz = anchors[:,2] * anchors[:,3], targets[:,2] * targets[:,3]
        union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter
        
        return inter/(union+1e-8)


    else: #circular anchors
        a, t = anchors.size(0), targets.size(0)
        ancs = anchors.unsqueeze(1).expand(a,t,3)
        tgts = targets.unsqueeze(0).expand(a,t,3)
        diff = (ancs[:,:,0:2]-tgts[:,:,0:2])
        distances = (diff**2).sum(dim=2).sqrt()
        radius1 = ancs[...,2]
        radius2 = tgts[...,2]
        acosterm1 = (((distances**2) + (radius1**2) - (radius2**2)) / (2 * distances * radius1)).clamp(-1,1).acos()
        acosterm2 = (((distances**2) - (radius1**2) + (radius2**2)) / (2 * distances * radius2)).clamp(-1,1).acos()
        secondterm = ((radius1+radius2-distances)*(distances+radius1-radius2)*(distances+radius1+radius2)*(distances-radius1+radius2)).clamp(min=0).sqrt()

        intersec = (radius1**2 * acosterm1) + (radius2**2 * acosterm2) - (0.5 * secondterm)

        union = np.pi * ((radius1**2) + (radius2**2)) - intersec

        return intersec / (union+1e-8)

    


def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    "Match `anchors` to targets. -1 is match to background, -2 is ignore."
    ious = IoU_values(anchors, targets)
    matches = anchors.new(anchors.size(0)).zero_().long() - 2

    if ious.shape[1] > 0:
        vals,idxs = torch.max(ious,1)
        matches[vals < bkg_thr] = -1
        matches[vals > match_thr] = idxs[vals > match_thr]
    #Overwrite matches with each target getting the anchor that has the max IoU.
    #vals,idxs = torch.max(ious,0)
    #If idxs contains repetition, this doesn't bug and only the last is considered.
    #matches[idxs] = targets.new_tensor(list(range(targets.size(0)))).long()
    return matches



        

def nms(boxes, scores, thresh=0.5):
    idx_sort = scores.argsort(descending=True)
    boxes, scores = boxes[idx_sort], scores[idx_sort]
    to_keep, indexes = [], torch.LongTensor(range_of(scores))
    while len(scores) > 0:
        #pdb.set_trace()
        to_keep.append(idx_sort[indexes[0]])
        iou_vals = IoU_values(boxes, boxes[:1]).squeeze()
        mask_keep = iou_vals <= thresh
        mask_keep_nonzero = torch.nonzero(mask_keep)
        if len(mask_keep_nonzero) == 0: break
        boxes, scores, indexes = boxes[mask_keep], scores[mask_keep], indexes[mask_keep]
    return LongTensor(to_keep)

def nms_obj(det:dict, thres=0.5):
    to_keep = nms(det['bbox_pred'], det['scores'], thresh=thres)
    for key in ['bbox_pred', 'preds', 'scores']:
        det[key] = det[key][to_keep]
    
    det['to_keep'] = to_keep
    return det
#    bbox_pred, preds, scores = det['bbox_pred'][to_keep], preds[to_keep].cpu(), scores[to_keep].cpu()
    
def dict_to_cpu(dictType:dict):
    for k in dictType.keys():
        if (dictType[k] is not None):
            dictType[k] = dictType[k].cpu()
    return dictType


   
            
def process_output(clas_pred, bbox_pred, anchors, detect_thresh=0.25, use_sigmoid=True):
    bbox_pred = activ_to_bbox(bbox_pred, anchors.to(clas_pred.device))

    if (use_sigmoid):
        clas_pred = torch.sigmoid(clas_pred)

    clas_pred_orig = clas_pred.clone()
    detect_mask = clas_pred.max(1)[0] > detect_thresh
    if np.array(detect_mask).max() == 0:
        return {'bbox_pred':None, 'scores':None, 'preds':None, 'clas_pred':clas_pred,'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}

    bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
    if (bbox_pred.shape[-1] == 4):
        bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))
    else:
        bbox_pred = bbox_pred# torch.clamp(bbox_pred, min=-1, max=1)

        
    scores, preds = clas_pred.max(1)
    return {'bbox_pred':bbox_pred, 'scores':scores, 'preds':preds, 'clas_pred':clas_pred, 'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}


def rescale_boxes(bboxes, t_sz: Tensor):
    t_sz = t_sz.to(bboxes.device)
    if (bboxes.shape[-1] == 4):
        bboxes[:, 2:] = bboxes[:, 2:] * t_sz / 2
        bboxes[:, :2] = (bboxes[:, :2] + 1) * t_sz / 2
    else:
        bboxes[:, 2:] = bboxes[:, 2:] * t_sz[...,0] / 2
        bboxes[:, :2] = (bboxes[:, :2] + 1) * t_sz / 2

    return bboxes



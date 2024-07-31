from .utils.nms_WSI import non_max_suppression_by_distance
from .utils.object_detection_helper import create_anchors
from .utils.inference_utils import DetectionInference
from .utils.models.RetinaNet import RetinaNetDA
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from typing import Callable
from tqdm import tqdm
import numpy as np
import logging
import zipfile
import torch
import os

update_steps = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)

from exact_sync.v1.models import PluginResultAnnotation, PluginResult, PluginResultEntry, Plugin, PluginJob

class MIDOG21Inference(DetectionInference):
    def __init__(self, **kwargs) -> None:
        super().__init__(down_factor = 1, patch_size = 512, mean=torch.FloatTensor([0.7481, 0.5692, 0.7225]), std=torch.FloatTensor([0.1759, 0.2284, 0.1792]),  detection_threshold = 0.4,  nms_threshold = 0.5, **kwargs)

    def configure_model(self):
        logging.info('Loading model')
        modelpath = os.path.join('QueueRunner', 'handlers', 'checkpoints', 'midog21.pth')
        scales = [0.2, 0.4, 0.6, 0.8, 1.0]
        ratios = [1]
        sizes = [(64, 64), (32, 32), (16, 16)]
        self.anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)

        encoder = create_body(resnet18(), pretrained=False, cut=-2)
        model = RetinaNetDA(encoder, n_classes=2, n_anchors=len(scales) * len(ratios),sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3, imsize=(512,512), n_domains=4)
        state_dict = torch.load(modelpath)
        state_dict = {key.replace('d5', 'discriminator'):value for key,value in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        return model
    
def inference(apis:dict, job:PluginJob, update_progress:Callable, **kwargs):

        image = apis['images'].retrieve_image(job.image)
        logging.info('Retrieving image set for job %d ' % job.id)


        update_progress(0.01)
        unlinklist=[] # files to delete
        imageset = image.image_set

        logging.info('Checking annotation type availability for job %d' % job.id)
        annotationtypes = {anno_type['name']:anno_type for anno_type in apis['manager'].retrieve_annotationtypes(imageset)}        
                    
        # The correct annotation type is required in order to be able to add the annotation
        # CAVE: The annotation type also needs to be a part of the product that you want to apply
        # the detection on.
        annoclass=None
        for t in annotationtypes:
            if 'MITOTIC FIGURE' in t.upper():
                annoclass = annotationtypes[t]
        
        if (annoclass is None):
            error_message = 'Error: Missing annotation type'
            error_detail = 'Annotation class Mitotic Figure is required but does not exist for imageset '+str(imageset)
            logging.error(str(error_detail))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False              
        

        try:
            tpath = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', image.filename)
            if not os.path.exists(tpath):
                if ('.mrxs' in str(image.filename).lower()):
                    tpath = tpath + '.zip'
                logging.info('Downloading image %s to %s' % (image.filename,tpath))
                apis['images'].download_image(job.image, target_path=tpath, original_image=False)
                if ('.mrxs' in str(image.filename).lower()):
                    logging.info('Unzipping MRXS image %s' % (tpath))

                    with zipfile.ZipFile(tpath, 'r') as zip_ref:
                        zip_ref.extractall('tmp/')
                        for f in zip_ref.filelist:
                            unlinklist.append('tmp/'+f.orig_filename)
                        unlinklist.append(tpath)
                    # Original target path is MRXS file
                    tpath = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', image.filename)
                    
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while downloading'
            error_detail = str(e)
            logging.error(str(e))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False            

        try:
            logging.info('Stage 1 for job %d' % job.id)
            inference_module = MIDOG21Inference(fname = tpath, update_progress = update_progress)
            stage1_results = inference_module.process()
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while processing stage 1'
            error_detail = str(e)
            logging.error(str(e))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

            
        try:
            if len(stage1_results)>0:
                logging.info('NMS after stage 1 for job %d ' % job.id)
                boxes = np.array(stage1_results)
                center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
                center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2  
                scores = boxes[:,4]
                stage1_results = non_max_suppression_by_distance(boxes=boxes, scores=scores, center_x=center_x, center_y=center_y).tolist()
                logging.info('NMS reduced stage1 results by %.2f percent.',  100*(1-(float(len(stage1_results))/boxes.shape[0])))
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while NMS.'
            error_detail = str(e)
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
            

        try:
            logging.info('Creating plugin result')
            existing = [j.id for j in apis['processing'].list_plugin_results().results if j.job==job.id]
            if len(existing)>0:
                apis['processing'].destroy_plugin_result(existing[0])
            
            # Create Result for job
            # Each job is linked to a single result, which may consist of several result entries.
            result = PluginResult(job=job.id, image=image.id, plugin=job.plugin, entries=[])
            result = apis['processing'].create_plugin_result(body=result)

            
            logging.info('Creating plugin entry')
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result'
            error_detail = str(e)+f'Job {job.id}, Image {image.id}, Pliugin {job.plugin}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
            
        try:
            # Create result entry for result
            # Each plugin result can contain collection of annotations. 
            resultentry = PluginResultEntry(pluginresult=result.id, name='Mitotic Figures', annotation_results = [], bitmap_results=[], default_threshold=0.4)
            resultentry = apis['processing'].create_plugin_result_entry(body=resultentry)
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result entry'
            error_detail = str(e)+f'PluginResult {result.id}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

        try:
            # Loop through all detections
            for n, line in enumerate(tqdm(stage1_results,desc='Uploading annotations (skip imposters)')):

                if (n%update_steps == 0):
                    update_progress (90+10*(n/len(stage1_results))) # 90.100% are for upload

                predcoords, score = line[0:4], line[4], 


                vector = {"x1": predcoords[0], "y1": predcoords[1], "x2": predcoords[2], "y2": predcoords[3]}

                anno = PluginResultAnnotation(annotation_type=annoclass['id'], pluginresultentry=resultentry.id, image=image.id, vector=vector, score=score)
                anno = apis['processing'].create_plugin_result_annotation(body=anno, async_req=True)
                    
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while uploading the annotations'
            error_detail = str(e)
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
        
        try:
            os.unlink(tpath)
            for f in unlinklist:
                os.unlink(f)
        
        except Exception as e:
            logging.error('Error while deleting files: '+str(e)+'. Continuing anyway.')
        
        return True


plugin = {  'name':'MIDOG 2021 baseline / 0.4 threshold',
            'author':'Frauke Wilm / Marc Aubreville', 
            'package':'science.imig.midog2021.baseline-da-lowthr', 
            'contact':'marc.aubreville@thi.de', 
            'abouturl':'https://github.com/DeepPathology/MIDOG_reference_docker', 
            'icon':'QueueRunner/handlers/logos/midog2021_logo.jpg',
            'products':[],
            'results':[],
            'inference_func' : inference}



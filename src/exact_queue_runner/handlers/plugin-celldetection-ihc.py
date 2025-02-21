from .utils.nms_WSI import non_max_suppression_by_distance
from .utils.object_detection_helper import create_anchors
from .utils.inference_utils import DetectionInference
from .utils.models.RetinaNet import RetinaNet
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

class LymphocyteInference(DetectionInference):
    def __init__(self, **kwargs) -> None:
        super().__init__(down_factor = 1, patch_size = 256, mean=torch.FloatTensor([0.7404, 0.7662, 0.7805]), std=torch.FloatTensor([0.1504, 0.1313, 0.1201]),  detection_threshold = 0.4,  nms_threshold = 0.5, **kwargs)

    def configure_model(self):
        logging.info('Loading model')
        modelpath = os.path.join('QueueRunner', 'handlers', 'checkpoints', 'pan_tumor.pth')
        scales=[0.5, 0.75, 1, 1.25, 1.5]
        ratios=[0.5, 1, 2]
        sizes=[(32, 32), (16, 16), (8, 8), (4, 4)]
        self.anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)

        encoder = create_body(resnet18(), pretrained=False, cut=-2)
        model = RetinaNet(encoder, n_classes=4, n_anchors=15, sizes=[32, 16, 8, 4], chs=128, final_bias=-4., n_conv=3)
        state_dict = torch.load(modelpath, map_location=torch.device('cpu'))['model']

        model.load_state_dict(state_dict)
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
        annoclasses={}
        match_dict = ['IMMUNE CELL', 'NON-TUMOR CELL', 'TUMOR CELL']
        for t in annotationtypes:
            for label_class in match_dict:
                if label_class == t.upper():
                    annoclasses[label_class] = annotationtypes[t]
        
        if (len(annoclasses.keys()) != 3):
            missing = list(set(match_dict) - set(annoclasses.keys()))
            error_message = 'Error: Missing annotation type(s)'
            error_detail = 'Annotation class {} is required but does not exist for imageset '.format(' and '.join(missing))+str(imageset)
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
            inference_module = LymphocyteInference(fname = tpath, update_progress = update_progress)
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
                stage1_results = non_max_suppression_by_distance(boxes=boxes, scores=scores, radius = 10, center_x=center_x, center_y=center_y).tolist()
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
            resultentry = PluginResultEntry(pluginresult=result.id, name='Cells', annotation_results = [], bitmap_results=[], default_threshold=0.64)
            resultentry = apis['processing'].create_plugin_result_entry(body=resultentry)
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result entry'
            error_detail = str(e)+f'PluginResult {result.id}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

        try:
            # Loop through all detections
            for n, line in enumerate(tqdm(stage1_results,desc='Uploading annotations')):

                if (n%update_steps == 0):
                    update_progress (90+10*(n/len(stage1_results))) # 90.100% are for upload

                predcoords, score = line[0:4], line[4]
                pred_class =  match_dict[int(line[5])]


                vector = {"x1": predcoords[0], "y1": predcoords[1], "x2": predcoords[2], "y2": predcoords[3]}

                anno = PluginResultAnnotation(annotation_type=annoclasses[pred_class]['id'], pluginresultentry=resultentry.id, image=image.id, vector=vector, score=line[4])
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

plugin = {  'name':'Cell Detection',
            'author':'Frauke Wilm', 
            'package':'science.imig.cell-det', 
            'contact':'frauke.wilm@fau.de', 
            'abouturl':'https://github.com/DeepMicroscopy/CD3-Detection', 
            'icon':'QueueRunner/handlers/logos/lymphocytes_logo.png',
            'products':[],
            'results':[],
            'inference_func' : inference}



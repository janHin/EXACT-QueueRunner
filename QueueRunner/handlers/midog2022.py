from .MIDOG2022.MIDOG2022_inference import inference as inference_MIDOG2022
from lib.nms_WSI import non_max_suppression_by_distance
import logging
from typing import Callable
import time
import os
from tqdm import tqdm

update_steps = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)

from exact_sync.v1.models import PluginResultAnnotation, PluginResult, PluginResultEntry, Plugin, PluginJob

def inference(apis:dict, job:PluginJob, update_progress:Callable, **kwargs):

        image = apis['images'].retrieve_image(job.image)
        logging.info('Retrieving image set for job %d ' % job.id)

        if ('.mrxs' in str(image.filename).lower()):
            logging.warning('Skipping MRXS file: '+image.filename)
            apis['processing'].partial_update_plugin_job(id=job.id, attached_worker=None)
            return False

        update_progress(0.01)
        
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
            tpath = 'tmp/'+image.filename
            if not os.path.exists(tpath):
                logging.info('Downloading image %s' % image.filename)
                apis['images'].download_image(job.image, target_path=tpath, original_image=False)
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while downloading'
            error_detail = str(e)
            logging.error(str(e))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False            

        try:
            logging.info('Stage 1 for job %d' % job.id)
            stage1_results = inference_MIDOG2022(tpath, update_progress=update_progress)
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while processing stage 1'
            error_detail = str(e)
            logging.error(str(e))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

            
        try:
            logging.info('NMS after stage 1 for job %d ' % job.id)
            boxes = np.array(stage1_results)
            center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2  
            scores = boxes[:,4]
            stage1_results = non_max_suppression_by_distance(boxes=boxes, scores=scores, center_x=center_x, center_y=center_y).tolist()
            logging.info('NMS reduced stage1 results by %.2f percent.',  100*(1-(float(len(stage1_results))/boxes.shape[0])))
        except Exception as e:
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
            
            
            # Create result entry for result
            # Each plugin result can contain collection of annotations. 
            resultentry = PluginResultEntry(pluginresult=result.id, name='Mitotic Figures', annotation_results = [], bitmap_results=[])
            resultentry = apis['processing'].create_plugin_result_entry(body=resultentry)


            # Loop through all detections
            for n, line in enumerate(tqdm(stage1_results,desc='Uploading annotations (skip imposters)')):

                if (n%update_steps == 0):
                    update_progress (90+10*(n/len(stage1_results))) # 90.100% are for upload

                predcoords, score = line[0:4], line[4], 


                vector = {"x1": predcoords[0], "y1": predcoords[1], "x2": predcoords[2], "y2": predcoords[3]}
                meta_data = f'score={score:.2f}'

                anno = PluginResultAnnotation(annotation_type=annoclass['id'], pluginresultentry=resultentry.id, image=image.id, vector=vector)
                anno = apis['processing'].create_plugin_result_annotation(body=anno)
                    
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while uploading the annotations'
            error_detail = str(e)
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
        
        
        os.unlink(tpath)
        return True


plugin = {  'name':'MIDOG 2022 Mitosis Domain Adversarial Baseline',
            'author':'Frauke Wilm / Marc Aubreville', 
            'package':'science.imig.midog2022.baseline-da', 
            'contact':'marc.aubreville@thi.de', 
            'abouturl':'https://github.com/DeepPathology/EXACT-QueueRunner/', 
            'icon':'handlers/MIDOG2022/midog_2022_logo.png',
            'products':[],
            'results':[],
            'inference_func' : inference}



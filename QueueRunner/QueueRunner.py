import config
import datetime
from MIDOG2022.MIDOG2022_inference import inference as inference_MIDOG2022
from lib.nms_WSI import non_max_suppression_by_distance
from tqdm import tqdm
import logging
import time
import numpy as np
import os

from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.processing_api import ProcessingApi
from exact_sync.v1.models import PluginResultAnnotation, PluginResult, PluginResultEntry, Plugin
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.exact_enums import *
from exact_sync.exact_errors import *
from exact_sync.exact_manager import *

update_steps = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)

configuration = Configuration()
configuration.username = config.username
configuration.password = config.password
configuration.host = config.serverurl

manager = ExactManager(username=config.username,   password=config.password, serverurl=config.serverurl, loglevel=100)
client = ApiClient(configuration=configuration)
processing_api = ProcessingApi(client)
images_api = ImagesApi(client)

plugin_id_midog2022 = None

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Find the plugin entry on EXACT
for plugin in processing_api.list_plugins().results:
    if plugin.package == 'science.imig.midog2022.baseline-da':
        plugin_id_midog2022=plugin.id
# or create it, if not found
if plugin_id_midog2022 is None:
    plugin_id_midog2022 = processing_api.create_plugin(name='MIDOG 2022 Mitosis Domain Adversarial Baseline',
                          author='Frauke Wilm / Marc Aubreville', 
                          package='science.imig.midog2022.baseline-da', 
                          contact='marc.aubreville@thi.de', 
                          abouturl='https://github.com/DeepPathology/EXACT-QueueRunner/', 
                          icon='MIDOG2022/midog_2022_logo.png',
                          products=[],
                          results=[]).id
                
import socket
import string
import random

worker_name = str(socket.gethostname()+'_'+''.join(random.choice(string.ascii_uppercase +
                                                string.ascii_lowercase +
                                                string.digits)
                                  for _ in range(6)))


logging.info('This is worker: '+worker_name)
logging.info(f'I am responsible for server plugin: {plugin_id_midog2022}')

while (True):
    jobs=processing_api.list_plugin_jobs(limit=100000000).results
    logging.info('Job queue contains '+str(len(jobs))+' jobs')
    for job in jobs:
        
        # Only work on jobs that are not already completed
        if (job.processing_complete==100):
            continue
        
        # MIDOG 2022 domain adversarial baseline
        if (job.plugin == plugin_id_midog2022) and (job.result is None) and (((job.attached_worker and (len(job.attached_worker)==0)) or (datetime.datetime.now()-job.updated_time).seconds>3600)):
            update_progress = lambda progress: processing_api.partial_update_plugin_job(id=job.id,processing_complete=progress, updated_time=datetime.datetime.now())

            processing_api.partial_update_plugin_job(id=job.id, attached_worker=worker_name)
            # re-check if we got the job after a random time below 1 second
            time.sleep(random.random())
            
            newjob = processing_api.retrieve_plugin_job(id=job.id)
            
            if (newjob.attached_worker != worker_name):
                logging.info('There was a conflict. Worker '+newjob.attached_worker+' got the job finally')
                continue

                
            logging.info('Successfully claimed job %d' % job.id)

            image = images_api.retrieve_image(job.image)

            if ('.mrxs' in str(image.filename).lower()):
                logging.warning('Skipping MRXS file: '+image.filename)
                processing_api.partial_update_plugin_job(id=job.id, attached_worker=None)
                continue

            update_progress(0.01)
            
            imageset = image.image_set

            logging.info('Retrieving image set for job %d ' % job.id)

            logging.info('Checking annotation type availability for job %d' % job.id)
            
            annotationtypes = {anno_type['name']:anno_type for anno_type in manager.retrieve_annotationtypes(imageset)}
            
                        
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
                processing_api.partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
                continue                
            

            try:
                tpath = 'tmp/'+image.filename
                if not os.path.exists(tpath):
                    logging.info('Downloading image %s' % image.filename)
                    images_api.download_image(job.image, target_path=tpath, original_image=False)
            except Exception as e:
                error_message = 'Error: '+str(type(e))+' while downloading'
                error_detail = str(e)
                logging.error(str(e))
                processing_api.partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
                continue            

            try:
                logging.info('Stage 1 for job %d' % job.id)
                stage1_results = inference_MIDOG2022(tpath, update_progress=update_progress)
            except Exception as e:
                error_message = 'Error: '+str(type(e))+' while processing stage 1'
                error_detail = str(e)
                logging.error(str(e))
                processing_api.partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
                continue

                
            try:
                logging.info('NMS after stage 1 for job %d ' % job.id)
                boxes = np.array(stage1_results)
                center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
                center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2  
                scores = boxes[:,4]
                stage1_results = non_max_suppression_by_distance(boxes=boxes, scores=scores, center_x=center_x, center_y=center_y).tolist()
                logging.info('NMS reduced stage1 results by %.2f percent.',  100*(1-(float(len(stage1_results))/boxes.shape[0])))
            except Exception as e:
                logging.error(str(e))
                import code
                code.interact(local=locals())
                

            try:
                logging.info('Creating plugin result')
                existing = [j.id for j in processing_api.list_plugin_results().results if j.job==job.id]
                if len(existing)>0:
                    processing_api.destroy_plugin_result(existing[0])
                
                # Create Result for job
                # Each job is linked to a single result, which may consist of several result entries.
                result = PluginResult(job=job.id, image=image.id, plugin=job.plugin, entries=[])
                result = processing_api.create_plugin_result(body=result)

                
                logging.info('Creating plugin entry')
                
                
                # Create result entry for result
                # Each plugin result can contain collection of annotations. 
                resultentry = PluginResultEntry(pluginresult=result.id, name='Mitotic Figures', annotation_results = [], bitmap_results=[])
                resultentry = processing_api.create_plugin_result_entry(body=resultentry)


                # Loop through all detections
                for n, line in enumerate(tqdm(stage1_results,desc='Uploading annotations (skip imposters)')):

                    if (n%update_steps == 0):
                        update_progress (90+10*(n/len(stage1_results))) # 90.100% are for upload

                    predcoords, score = line[0:4], line[4], 


                    vector = {"x1": predcoords[0], "y1": predcoords[1], "x2": predcoords[2], "y2": predcoords[3]}
                    meta_data = f'score={score:.2f}'

                    anno = PluginResultAnnotation(annotation_type=annoclass['id'], pluginresultentry=resultentry.id, image=image.id, vector=vector)
                    anno = processing_api.create_plugin_result_annotation(body=anno)
                        
            except Exception as e:
                error_message = 'Error: '+str(type(e))+' while uploading the annotations'
                error_detail = str(e)
                logging.error(str(e))
                import code
                code.interact(local=locals())
                
                processing_api.partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
                continue
            
            
            update_progress(100.0)
            os.unlink(tpath)
            processing_api.partial_update_plugin_job(id=job.id, attached_worker=None)

            logging.info('Unclaimed job %d' % job.id)                
                
 

    time.sleep(5)
    

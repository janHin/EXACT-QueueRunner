#STL imports
import os
import gc

#3rd party imports
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from typing import Callable
from pathlib import Path
from tqdm import tqdm
import logging
import zipfile
import joblib
import torch
from exact_sync.v1.models import PluginResultAnnotation, PluginResult, PluginResultEntry, Plugin, PluginJob

#local imports
from .utils.inference_utils import DetectionInference

update_steps = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)


class NucleusInference(DetectionInference):
    def __init__(self, **kwargs) -> None:
        super().__init__(down_factor = 1, patch_size = 512, mean=None, std=None,  detection_threshold = 0.55, **kwargs)

    def configure_model(self):
        logging.info('Loading model')
        model = NucleusInstanceSegmentor(
            pretrained_model="hovernet_fast-pannuke",
            num_loader_workers=0,
            num_postproc_workers=0,
            batch_size=8,
            auto_generate_mask=False,
            verbose=False,
        )

        return model
    
    def process(self):
        self.model = self.configure_model()
        mpp = float(self.slide.properties['openslide.mpp-x'])
        
        wsi_output = self.model.predict(
            [self.slide._filename],
            masks=None,
            save_dir="QueueRunner/tmp/{}/".format(Path(self.slide._filename).stem),
            mode="wsi",
            on_gpu=True if str(self.device) == 'cuda' else False,
            crash_on_exception=True,
        )

        wsi_pred = joblib.load(f"{wsi_output[0][1]}.dat")
        logging.info("Number of detected nuclei: %d", len(wsi_pred))
        # free up memory
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.update_progress(95)
        wsi_pred = {key:{'box': value['box']*(0.25/mpp), 'prob': value['prob']} for key, value in wsi_pred.items()}
        return wsi_pred

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
            if 'NUCLEUS' in t.upper():
                annoclass = annotationtypes[t]
        
        if (annoclass is None):
            error_message = 'Error: Missing annotation type'
            error_detail = 'Annotation class Nucleus is required but does not exist for imageset '+str(imageset)
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
            logging.info('Prediction for job %d' % job.id)
            inference_module = NucleusInference(fname = tpath, update_progress = update_progress)
            stage1_results = inference_module.process()
            #unlinklist.append(os.path.join(os.getcwd(), 'QueueRunner', 'tmp', Path(image.filename).stem))
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while processing WSI'
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
            resultentry = PluginResultEntry(pluginresult=result.id, name='Nucleus', annotation_results = [], bitmap_results=[], default_threshold=0.5)
            resultentry = apis['processing'].create_plugin_result_entry(body=resultentry)
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result entry'
            error_detail = str(e)+f'PluginResult {result.id}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

        try:
            # Loop through all detections
            for n, key in enumerate(tqdm(stage1_results,desc='Uploading annotations')):

                if (n%update_steps == 0):
                    update_progress (90+10*(n/len(stage1_results))) # 90.100% are for upload

                line = stage1_results[key]
                predcoords, score = line["box"], line["prob"], 

                vector = {"x1": float(predcoords[0]), "y1": float(predcoords[1]), "x2": float(predcoords[2]), "y2": float(predcoords[3])}
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


plugin = {  'name':'HoverNet Nucleus Detection',
            'author':'Frauke Wilm', 
            'package':'science.imig.hovernet', 
            'contact':'frauke.wilm@fau.de', 
            'abouturl':'https://github.com/TissueImageAnalytics/tiatoolbox/', 
            'icon':'QueueRunner/handlers/logos/hovernet_logo.png',
            'products':[],
            'results':[],
            'inference_func' : inference}



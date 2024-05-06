from .utils.inference_utils import SegmentationInference
from fastai.vision.models.unet import DynamicUnet
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from typing import Callable
from pathlib import Path
import logging
import zipfile
import torch
import h5py
import os

update_steps = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)

from exact_sync.v1.models import PluginResultBitmap, PluginResult, PluginResultEntry, Plugin, PluginJob, Image

class CATCHInference(SegmentationInference):
    def __init__(self, **kwargs) -> None:
        super().__init__(down_factor = 16, patch_size = 512, mean=torch.FloatTensor([0.7587, 0.5718, 0.6572]), std=torch.FloatTensor([0.0866, 0.1118, 0.0990]), **kwargs)

    def configure_model(self):
        logging.info('Loading model')
        modelpath = os.path.join('QueueRunner', 'handlers', 'checkpoints', 'catch.pth')
        encoder = create_body(resnet18(), n_in=3, pretrained=True, cut=-2)
        model = DynamicUnet(encoder, n_out=6, img_size=(self.patch_size,self.patch_size))
        model_keys = list(model.state_dict().keys())
        checkpoint_dict = torch.load(modelpath, map_location=self.device)["model"]
        state_dict = {}
        for k, (key, value) in enumerate(checkpoint_dict.items()):
            state_dict[model_keys[k]] = value
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
        match_dict = ['TUMOR', 'EPIDERMIS', 'DERMIS', 'SUBCUTIS', 'INFLAMM/NECROSIS']
        for t in annotationtypes:
            for label_class in match_dict:
                if label_class in t.upper():
                    annoclasses[label_class] = annotationtypes[t]
        
        missing = list(set(match_dict) - set(annoclasses.keys()))
        if len(missing) > 0:
            warning = 'Warning: Missing annotation type(s)'
            error_detail = 'Annotation class {} does not exist for imageset '.format(' and '.join(missing))+str(imageset)
            logging.warning(str(error_detail))

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
            logging.info('Inference for job %d' % job.id)
            hdf5_path = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', "{}.hdf5".format(Path(image.filename).stem))
            hdf5_file = h5py.File(hdf5_path, "w")
            inference_module = CATCHInference(fname = tpath, update_progress = update_progress, hdf5_file=hdf5_file)
            segmentation_results = inference_module.process()
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while processing segmentation inference'
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
            resultentry = PluginResultEntry(pluginresult=result.id, name='Segmentation', annotation_results = [], bitmap_results=[], default_threshold=0.0) # optionally set threshold
            resultentry = apis['processing'].create_plugin_result_entry(body=resultentry)
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result entry'
            error_detail = str(e)+f'PluginResult {result.id}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

        try:
            image_type = int(Image.ImageSourceTypes.DEFAULT)
            for mask_path in segmentation_results:
                image = apis['images'].create_image(file_path=mask_path, image_type=image_type, image_set=imageset).results[0]
                unlinklist.append(mask_path)
           
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while uploading the segmentations'
            error_detail = str(e)
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
        
        try:
            os.unlink(tpath)
            os.unlink(hdf5_path)
            for f in unlinklist:
                os.unlink(f)
        
        except Exception as e:
            logging.error('Error while deleting files: '+str(e)+'. Continuing anyway.')
        
        return True


plugin = {  'name':'CATCH segmentation baseline',
            'author':'Frauke Wilm', 
            'package':'science.imig.catch', 
            'contact':'frauke.wilm@fau.de', 
            'abouturl':'https://github.com/DeepPathology/EXACT-QueueRunner', 
            'icon':'QueueRunner/handlers/logos/catch_logo.jpg',
            'products':[],
            'results':[],
            'inference_func' : inference}



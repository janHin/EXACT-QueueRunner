#STL imports
import os
import gc
from pathlib import Path
import abc
import atexit

#3rd party imports
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from typing import Callable
from pathlib import Path
from tqdm import tqdm
import logging
import zipfile
import joblib
import torch
from exact_sync.v1.models import (PluginResultAnnotation, PluginResult,
    PluginResultEntry, Plugin, PluginJob, Image)

#local imports
from .utils.inference_utils import DetectionInference
from .utils.exception import PluginExcpetion

UPDATE_STEPS = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)

logger = logging.getLogger(__name__)

class NucleusInference(DetectionInference):
    def __init__(self, outdir:Path,**kwargs) -> None:
        super().__init__(down_factor = 1, patch_size = 512, mean=None, std=None,
            detection_threshold = 0.55, **kwargs)
        
        if not outdir.is_dir():
            raise FileNotFoundError(f'could not find outdir: {outdir}')
        self.outdir = outdir

        self._mpp = float(self.slide.properties['openslide.mpp-x'])

    def configure_model(self):
        logger.info('Loading model')
        model = NucleusInstanceSegmentor(
            pretrained_model="hovernet_fast-pannuke",
            num_loader_workers=0,
            num_postproc_workers=0,
            batch_size=8,
            auto_generate_mask=False,
            verbose=False,
        )

        return model

    def process(self)->Path:
        self.model = self.configure_model()
        

        save_dir = self.outdir / Path(self.slide._filename).stem
        wsi_output = self.model.predict(
            [self.slide._filename],
            masks=None,
            save_dir=save_dir,
            mode="wsi",
            device=self.device.type,
            crash_on_exception=True,
        )

        # free up memory
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.update_progress(95)

        return save_dir / f"{wsi_output[0][1]}.dat"
    
    def load_predictions(self,outfile:Path)->dict:

        wsi_pred = joblib.load(outfile)
        logger.info("Number of detected nuclei: %d", len(wsi_pred))

        wsi_pred = {key:
            {'box': value['box']*(0.25/self._mpp), 'prob': value['prob']}
            for key, value in wsi_pred.items()
            }
        return wsi_pred

class Plugin(abc.ABC):
    
    def __init__(self,apis:dict,update_progress:Callable=None) -> None:
        super().__init__()
        self.apis = apis
        self.update_progress_func = update_progress

        atexit.register(self._cleanup)

        self._unlinklist = []

    def _cleanup(self):
        logger.info('cleaning up files')
        for path in self._unlinklist:
            if isinstance(str,path):
                path = Path(path)
            logger.debug(f'deleting path {path}')
            path.unlink(missing_ok=True)

    def __delattr__(self, name: str) -> None:
        return super().__delattr__(name)

    def unlink_path(self,path:Path):
        self._unlinklist.append(path)



    @abc.abstractmethod
    def do_inference(self,job:PluginJob):
        pass

    def inference(self,job:PluginJob):
        try:
            self.do_inference(job)
        except PluginExcpetion as e:
            self.apis['processing'].partial_update_plugin_job(id=job.id,
                error_message=e.message, error_detail=e.detail)
            raise e
        except Exception as e:
            error_message=f'Exception {str(type(e))}'
            self.apis['processing'].partial_update_plugin_job(id=job.id,
                error_message=error_message, error_detail=str(e))
            raise e
    
class HovernetPlugin(Plugin):

    def __init__(self, apis: dict,update_progress:Callable,outdir:Path) -> None:
        super().__init__(apis,update_progress)
        self.outdir = outdir

        self.annoationtype = None        
        self.image         = None
        self.image_file    = None


    def _setup_data(self,job:PluginJob,error_image_exists:bool=True):
        logger.info('Retrieving image set for job %d ', job.id)
        self.image = self.apis['images'].retrieve_image(job.image)
        self.annoationtype = self._get_annotationtype(self.image.image_set)
        self.image_file = self._download_image(error_exists=error_image_exists)

    def do_inference(self,job:PluginJob):
        self.update_progress_func(0.01)
        self._setup_data(job,error_image_exists=True)
        self.update_progress_func(0.05)

        inference_module = NucleusInference(self.outdir,fname = self.image_file,
            update_progress = self.update_progress_func)
        inference_results_file = inference_module.process()
        inference_results      = inference_module.load_predictions(inference_results_file)


        self._upload_job_results(job,inference_results)

    def continue_inference(self,job:PluginJob):
        '''try to see whats left on an aborted job and upload existing results'''
        self._setup_data(job,error_image_exists=False)

        #TODO: still some hardcoding in here
        inference_module = NucleusInference(self.outdir,fname = self.image_file,
            update_progress = self.update_progress_func)
        inference_results_file = inference_module.outdir / self.image_file.stem / '0.dat'
        if not inference_results_file.is_file():
            raise FileNotFoundError('could not find inference file '
                f'{inference_results_file} for job {job.id}')
        inference_results = inference_module.load_predictions(inference_results_file)

        self._upload_job_results(job,inference_results)

    def _get_annotationtype(self,imageset:int)->dict:

        annotationtypes = {
            anno_type['name']:anno_type
            for anno_type in self.apis['manager'].retrieve_annotationtypes(imageset)
        }

        def filterfunc(annotationtye_name:str):
             if 'NUCLEUS' in annotationtye_name.upper():
                return True
             return False

        annotationtypes_filtered = {
            name:value
            for name,value in annotationtypes.items()
            if filterfunc(name)
        }

        if len(annotationtypes)<=0:
            error_message = 'Error: Missing annotation type'
            error_detail = ('Annotation class Nucleus is required but does not exist'
                ' for imageset '+str(imageset))
            raise PluginExcpetion(error_message,error_detail)

        annoclass = annotationtypes_filtered.values()[0]
        return annoclass

    def _download_image(self,error_exists:bool=True)->Path:

        if self.image is None:
            raise RuntimeError('Internal Error: something went wrong, '
                'self.image is none ')

        tpath = self.outdir / self.image.filename
        if tpath.exists():
            if error_exists:
                raise FileExistsError(f'file {tpath} exists already')
            return tpath
        # if ('.mrxs' in str(image.filename).lower()):
        #     tpath = tpath.with_suffix('.zip')
        logger.info('Downloading image %s to %s',str(self.image.filename),
            str(tpath))
        self.apis['images'].download_image(self.image.id, target_path=tpath,
            original_image=False)

        if ('.mrxs' in str(self.image.filename).lower()):
            logger.info('Unzipping MRXS image %s',tpath)

            with zipfile.ZipFile(tpath, 'r') as zip_ref:
                zip_ref.extractall('tmp/')
                for f in zip_ref.filelist:
                    self.unlink_path('tmp/'+f.orig_filename)
                self.unlink_path(tpath)
            # Original target path is MRXS file
            tpath = self.outdir / self.image.filename
        return tpath

    def _remove_existing_results(self,job:PluginJob):
        ''''''
        logger.info('checking for already existing results')
        existing_results = [
            result.id for result in self.apis['processing'].list_plugin_results().results
            if result.job==job.id
        ]
        logger.info('found %d already existing results. Start deleting',
            len(existing_results))
        for existing_result in existing_results:
            logger.debug('deleting job result (%d)',existing_result)
            self.apis['processing'].destroy_plugin_result(existing_result)

    def _upload_job_results(self,job:PluginJob, inference_results:dict):
        ''''''
        self._remove_existing_results(job)

        # Create Result for job
        # Each job is linked to a single result, which may consist of several result entries.
        logger.info('creating plugin result')
        result = PluginResult(job=job.id, image=self.image.id, plugin=job.plugin, entries=[])
        result = self.apis['processing'].create_plugin_result(body=result)

        logger.info('Creating plugin result entry')
        #----------Create result entries---------
        # Create result entry for result
        # Each plugin result can contain collection of annotations.
        resultentry = PluginResultEntry(pluginresult=result.id, name='Nucleus',
            annotation_results = [], bitmap_results=[], default_threshold=0.5)
        resultentry = self.apis['processing'].create_plugin_result_entry(body=resultentry)

        #----------Upload results-------------
        for n, key in enumerate(tqdm(inference_results,desc='Uploading annotations')):

            if (n%UPDATE_STEPS == 0):
               self.update_progress_func (90+10*(n/len(inference_results))) # 90.100% are for upload

            line = inference_results[key]
            predcoords, score = line["box"], line["prob"],

            vector = {
                    "x1": float(predcoords[0]),
                    "y1": float(predcoords[1]),
                    "x2": float(predcoords[2]),
                    "y2": float(predcoords[3])
                    }
            anno = PluginResultAnnotation(annotation_type=self.annoationtype['id'],
                pluginresultentry=resultentry.id,
                image=self.image.id,
                vector=vector,
                score=score)
            anno = self.apis['processing'].create_plugin_result_annotation(
                body=anno, async_req=True)

       
def entrypoint(apis:dict, job:PluginJob, update_progress:Callable,
        outdir:Path = None)->bool:
    
    if outdir is None:
        outdir = Path.cwd() / 'QueueRunner/tmp'
        if not outdir.is_dir():
            outdir.mkdir(parents=True)
    
    plugin = HovernetPlugin(apis,update_progress,outdir)
    plugin.inference(job)


plugin = {  'name':'HoverNet Nucleus Detection',
            'author':'Frauke Wilm', 
            'package':'science.imig.hovernet', 
            'contact':'frauke.wilm@fau.de', 
            'abouturl':'https://github.com/TissueImageAnalytics/tiatoolbox/', 
            'icon':'QueueRunner/handlers/logos/hovernet_logo.png',
            'products':[],
            'results':[],
            'inference_func' : entrypoint,
            'class': HovernetPlugin}


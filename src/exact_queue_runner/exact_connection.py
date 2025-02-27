#STL imports
import logging
import time
import datetime
from   typing import List

#exact imports
from exact_sync.v1.configuration import Configuration

from exact_sync.v1.api_client         import ApiClient
from exact_sync.v1.api.processing_api import ProcessingApi
from exact_sync.v1.api.images_api     import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi

from exact_sync.v1.models import (PluginResultAnnotation, PluginResult,
    PluginResultEntry, Plugin,PluginJob, Image,Images,ImageSet,ImageSets)

from exact_sync.exact_manager import ExactManager

from exact_sync.v1.rest import ApiException

#local imports

logger = logging.getLogger(__name__)

class JobRemovedException(Exception):
    pass

class ExactConnection():

    def __init__(self,configuration:Configuration) -> None:
        self._manager        = ExactManager(username=configuration.username,   
            password=configuration.password, serverurl=configuration.host,
            loglevel=100)
        self._client         = ApiClient(configuration=configuration)
        self._processing_api = ProcessingApi(self._client)
        self._images_api     = ImagesApi(self._client)
        self._imageset_api    = ImageSetsApi(self._client)

    @property
    def api_dict(self):
        return {
            'images':     self._images_api,
            'processing': self._processing_api,
            'manager' :   self._manager,
        }

    def get_exact_plugins(self):
        plugins_exact = {plugin.package: plugin for plugin in
            self._processing_api.list_plugins().results}
        return plugins_exact

    def register_plugins(self,plugins):
        '''register plugins with exact'''
        plugins_exact = self.get_exact_plugins()
        for k in plugins:
            plugin = plugins[k].plugin

            if (plugin['package'] not in plugins_exact):
                # Missing plugin on EXACT, let's register it.
                plugin_entries = {k:v for k,v in 
                    zip(plugin.keys(),plugin.values()) if k != 'inference_func'}

                self._processing_api.create_plugin(**plugin_entries)

    def retrieve_job(self,job_id:int)->PluginJob:
        job = self._processing_api.retrieve_plugin_job(job_id)
        return job

    @staticmethod
    def is_valid_job(job:PluginJob)->bool:

        if job.error_message:
            logger.warning('job (%d) has error: %s \n continuing',job.id,
                str(job.error_message))
            return False

        if job.processing_complete >= 100:
            logger.warning('job (%d) already has progress 100%%',job.id)
            return False

        #if job.result is not None:
        #    logger.info('job (%d) already has result attached',job.id)
        #    return False

        if job.attached_worker is not None and (len(job.attached_worker)>0):
            logger.info('job (%d) already has worker attached',job.id)
            return False
    
        return True


    def get_next_job(self,timeout:float=120)->PluginJob:
        ''''''
        start_time = datetime.datetime.now()
        while True:
            jobs=self._processing_api.list_plugin_jobs(limit=1e3).results
            # Only work on jobs that are not already completed or failed with an error
            logger.info('received %d jobs from server',len(jobs))
            
            try:
                job = next(filter(ExactConnection.is_valid_job,jobs))
                self.retrieve_job(job.id)
                return job
            except StopIteration:
                pass
            seconds_passed = (datetime.datetime.now() - start_time).seconds
            if seconds_passed > timeout:
                raise TimeoutError('timeout while looking for new jobs')
            
        return None

    def destroy_job(self,job_id:int):
        ''''''
        try:
            job = self._processing_api.retrieve_plugin_job(job_id,async_req=False)
        except ApiException as exc:
            if exc.status == 404:
                raise JobRemovedException(f'job ({job_id}) not found') from exc      
            raise exc
        self._processing_api.destroy_plugin_job(id=job_id,async_req=False)
        time.sleep(.5)
        try:
            job = self._processing_api.retrieve_plugin_job(job_id,async_req=False)
        except ApiException as exc:
            if exc.status != 404:
                raise exc

    def get_image_set(self,name:str)->ImageSet:
        ''''''
        logger.info('getting imageset')
        imagesets = self._imageset_api.list_image_sets(async_req=False).results
        imagesets_filtered = [ims for ims in imagesets if ims.name == name]
        if len(imagesets_filtered) <= 0:
            raise KeyError(f'could not find image set with name {name}')
        if len(imagesets_filtered) > 1:
            raise KeyError(f'found multiple image sets with name {name}')
        return imagesets_filtered[0]
        
    def get_images(self,name:str=None,image_set:int|str=None)->List[Image]:
        ''''''
        logger.info('getting images')
        
        if isinstance(image_set,int):
            image_set_id = image_set
        else:
            image_set_id = self.get_image_set(image_set).id
        logger.info('image set id: %d',image_set_id)

        images = self._images_api.list_images(async_req=False,name=name,
            image_set=image_set_id).results
        logger.info('images %s',str(images))

        return images


    def destroy_results_for_imageid(self,image_id:int):
        ''''''
        plugin_results = self._processing_api.list_plugin_results(
            async_req=False,image_id=image_id).results

        if len(plugin_results) <= 0:
            raise KeyError('found no entriee in plugin results for image id '
                f'{image_id}')
        if len(plugin_results) >1:
            raise KeyboardInterrupt('found multiple entries for image id '
                f'{image_id} in plugin results')

        plugin_result_id = plugin_results[0].id
        logger.info('deleting plugin result with id %d',plugin_result_id)
        self._processing_api.destroy_plugin_result(plugin_result_id,async_req=False)

    def update_job_exception(self,job:PluginJob,exception:Exception):
        ''''''
        error_message = f'Exception: {str(type(exception))}'
        error_detail = str(exception)
        self._processing_api.partial_update_plugin_job(id=job.id,
            error_message=error_message, error_detail=error_detail,
            async_req=False)

    def update_job_progress(self,job:PluginJob,progress:float):
        ''''''
        self._processing_api.partial_update_plugin_job(id=job.id,
            processing_complete=progress, 
            updated_time=datetime.datetime.now())
    
    def update_job_worker(self,job:PluginJob,worker:str):
        ''''''
        self._processing_api.partial_update_plugin_job(id=job.id,
            attached_worker=worker, updated_time=datetime.datetime.now())

    def update_job_released(self,job_id:PluginJob|int):
        ''''''
        if isinstance(job_id,PluginJob):
            job_id = job_id.id
        self._processing_api.partial_update_plugin_job(id=job_id,
            error_message=None, error_detail=None, attached_worker=None)

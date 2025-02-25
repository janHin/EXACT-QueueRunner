#STL imports
import datetime
import logging
import pkgutil
import importlib
import time
import socket
import string
import random
from dataclasses import dataclass
from typing import List,Any

#3rd party imports
import numpy as np

import h5py

#exact imports
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.processing_api import ProcessingApi
from exact_sync.v1.models import (PluginResultAnnotation, PluginResult,
    PluginResultEntry, Plugin,PluginJob, Image,Images,ImageSet,ImageSets)
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.exact_enums import *
from exact_sync.exact_errors import *
from exact_sync.exact_manager import *
from exact_sync.v1.rest import ApiException

#local imports
from . import handlers
from .utils import iter_namespace, get_workername

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



    def get_next_job(self)->PluginJob:
        ''''''
        jobs=self._processing_api.list_plugin_jobs(limit=1e3).results
        # Only work on jobs that are not already completed or failed with an error
        unprocessed_jobs = [job for job in jobs if job.processing_complete != 100]
        logger.info('Job queue contains '+str(len(unprocessed_jobs))+' unprocessed jobs')

        for job in unprocessed_jobs:
            # Get update about job
            try:
                job = self._processing_api.retrieve_plugin_job(id=job.id)
            except:
                logger.warning('Job unexpectedly removed from queue: '+str(job.id))
                continue
            return job
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
        images = self._images_api.list_images(async_req=False).results

        if isinstance(image_set,int):
            image_set_id = image_set
        else:
            image_set_id = self.get_image_set(image_set).id

        def filter_func(image:Image)->bool:
            logger.info('image %s',str(image))
            if image_set_id is not None and image.image_set != image_set_id:
                return False
            if name is not None and image.filename != name:
                return False
            return True

        images = [img for img in images if filter_func(img)]
        logger.info('filtered images %s',images)
        return images

    def get_plugin_results(self)->List[PluginResult]:
        ''''''
        plugin_results = self._processing_api.list_plugin_results(asnyc_req=False).results
        return plugin_results

    def destroy_results_for_imageid(self,image_id:int):
        ''''''
        plugin_results = self.get_plugin_results()

        plugin_results_filtered = [plr for plr in plugin_results 
            if plr.image == image_id]

        if len(plugin_results_filtered) <= 0:
            raise KeyError('found no entriee in plugin results for image id '
                f'{image_id}')
        if len(plugin_results_filtered) >1:
            raise KeyboardInterrupt('found multiple entries for image id '
                f'{image_id} in plugin results')

        plugin_result_id = plugin_results_filtered[0].id
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

    def update_job_released(self,job:PluginJob):
        ''''''
        self._processing_api.partial_update_plugin_job(id=job.id,
            error_message=None, error_detail=None, attached_worker=None)

class PluginHandler():

    def __init__(self,exact_connection:ExactConnection) -> None:
        self._local_plugins = PluginHandler.get_local_plugins()
        self._exact_plugins = exact_connection.get_exact_plugins()

    @staticmethod
    def get_local_plugins():
        '''get local plugin modules from handlers subfolder'''
        plugins={}

        for finder, name, ispkg in sorted(iter_namespace(handlers)):
            try:
                logger.info('activating plugin %s',name)
                mod = importlib.import_module(name)
                plugins[name] = mod.plugin
            except Exception as e:
                raise RuntimeError('+++ Unable to activate plugin: '+name) from e
        return  plugins

    def get_plugin_for_job(self,job:PluginJob):
        for plugin in self._local_plugins.values():
            if job.plugin == self._exact_plugins[plugin['package']].id:
                return plugin
        return None


def is_valid_job(job:PluginJob)->bool:

    if job.error_message:
        logger.warning('job (%d) has error: %s \n continuing',job.id,
            str(job.error_message))
        return False

    if job.result is not None:
        logger.info('job (%d) already has result attached',job.id)
        return False

    if job.attached_worker is not None and (len(job.attached_worker)>0):
        logger.info('job (%d) already has result attached',job.id)
        return False
    return True

def process_job(exact_connection:ExactConnection,job:PluginJob,plugin,
    outdir:Path)->bool:
    '''
    '''

    def update_progress(progress:float):
        exact_connection.update_job_progress(job,progress)

    try:
        success = plugin['inference_func'](apis=exact_connection.api_dict, job=job,
            update_progress=update_progress,outdir=outdir)
        if not success:
            raise RuntimeError(f'encountered error running plugin {plugin["name"]}')
    except Exception as e:
        logger.error('encountered error (%s) running inference_func'
            ' for %s',str(e),plugin['name'])
        exact_connection.update_job_exception(job,e)
        raise e

    exact_connection.update_job_progress(job,100.0)

def do_run(exact_connection:ExactConnection,plugin_handler:PluginHandler,
        worker_name:str,outdir:Path)->bool:
    ''''''

    job = exact_connection.get_next_job()
    
    if job is None:
        logger.info("no job returned by get_job()")
        return None

    if not is_valid_job(job):
        logger.info("job %s not valid",str(job.id))
        return False

    plugin = plugin_handler.get_plugin_for_job(job)

    if plugin is None:
        logger.info('no plugin to process job %s (plugin_id:%s)',
            str(job.id),str(job.plugin))
        return False

    logger.info('Job %s: Last update for this job was: %s seconds ago',
        str(job.id),
        str((datetime.datetime.now()-job.updated_time).seconds))
    
    logger.info('Job %s: Attached worker info: %s',str(job.id),
        str(job.attached_worker))
    logger.info(f'Job {job.id}: Claiming job.')
    
    exact_connection.update_job_worker(job,worker_name)

    # re-check if we got the job after a random time below 1 second
    time.sleep(random.random())
    job = exact_connection._processing_api.retrieve_plugin_job(id=job.id)
        
    if (job.attached_worker != worker_name):
        logger.info('There was a conflict. Worker %s got the job finally',
            str(job.attached_worker))
        return False
    
    logger.info('Claiming was: %.2f seconds ago.',
        (datetime.datetime.now()-job.updated_time).seconds)
    logger.info('Successfully claimed job %d' % job.id)

    try:
        process_job(exact_connection,job,plugin,outdir)
        exact_connection.update_job_released(job)
    except Exception as e:

        raise e from e
        

    logger.info('unclaiming job %d' % job.id)     
    # Break for loop to achieve refreshing of jobs list

def run_loop(exact_connection:ExactConnection,job_limit:int=-1,
    restart:bool=True,idle_limit:float=-1,outdir:Path=None):

    time.sleep(np.random.randint(5))

    plugin_handler = PluginHandler(exact_connection)

    worker_name = get_workername()
    logger.info('This is worker: '+worker_name)

    n_jobs = 0
    idle_time = 0.
    start_time = datetime.datetime.now()

    try:
        while (True):
            do_run(exact_connection,plugin_handler,worker_name,outdir)

            n_jobs += 1

            if job_limit >0 and n_jobs >= job_limit:
                break

            idle_time = (datetime.datetime.now() - start_time).seconds

            if idle_limit > 0 and idle_time > idle_limit:
                logger.info('idle time (%ds) exceeds limit (%ds)',idle_time,
                    idle_limit)

            time.sleep(5)

    except Exception as e:
        # restart
        logging.error('Caught exception.',exc_info=True)
        if restart:
            run_loop()
        raise e

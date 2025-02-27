#STL imports
import datetime
import logging
import time
import random
from pathlib import Path

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
from exact_sync.v1.rest import ApiException

#local imports
from .utils import get_workername
from .exact_connection import ExactConnection
from .plugin_handler import PluginHandler

logger = logging.getLogger(__name__)


def do_run(exact_connection:ExactConnection,plugin_handler:PluginHandler,
        worker_name:str,outdir:Path,keep_inputs:bool)->bool:
    ''''''

    job = exact_connection.get_next_job()

    if job is None:
        logger.info("no job returned by get_job()")
        raise RuntimeError('no job received')

    # if not is_valid_job(job):
    #     logger.info("job %s not valid",str(job.id))
    #     return False

    plugin_type = plugin_handler.get_plugin_for_job(job)

    if plugin_type is None:
        logger.info('no plugin to process job %s (plugin_id:%s)',
            str(job.id),str(job.plugin))
        return False

    plugin_instance = plugin_type(exact_connection,outdir,keep_inputs)

    logger.info('Job %s: Last update for this job was: %s seconds ago',
        str(job.id),
        str((datetime.datetime.now()-job.updated_time).seconds))

    logger.info('Job %s: Attached worker info: %s',str(job.id),
        str(job.attached_worker))
    logger.info('Job %d: Claiming job.',job.id)

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
    logger.info('Successfully claimed job %d', job.id)

    plugin_instance.inference(job)
    #process_job(exact_connection,job,plugin,outdir)
    exact_connection.update_job_progress(job,100.0)
    logger.info('unclaiming job %d', job.id)
    exact_connection.update_job_released(job)

def run_loop(exact_connection:ExactConnection,job_limit:int=-1,
    restart:bool=True,idle_limit:float=-1,outdir:Path=None,
    keep_inputs:bool=False):

    time.sleep(np.random.randint(5))

    plugin_handler = PluginHandler(exact_connection)

    worker_name = get_workername()
    logger.info('This is worker: %s',worker_name)

    n_jobs = 0
    idle_time = 0.
    start_time = datetime.datetime.now()

    try:
        while (True):
            do_run(exact_connection,plugin_handler,worker_name,outdir,keep_inputs)

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
            run_loop(exact_connection, job_limit, restart, idle_limit, outdir)
        raise e

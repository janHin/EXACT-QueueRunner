import config
import datetime
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

configuration = Configuration()
configuration.username = config.username
configuration.password = config.password
configuration.host = config.serverurl

manager = ExactManager(username=config.username,   password=config.password, serverurl=config.serverurl, loglevel=100)
client = ApiClient(configuration=configuration)
processing_api = ProcessingApi(client)
images_api = ImagesApi(client)

import pkgutil
import importlib
import handlers as handlers

def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

plugins={}

for finder, name, ispkg in sorted(iter_namespace(handlers)):
    try:
        mod = importlib.import_module(name)
        plugins[name] = mod
    except Exception as e:
        print('+++ Unable to active plugin: '+name,e)
        pass

plugin_id_midog2022 = None

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

plugins_exact = {plugin.package: plugin for plugin in processing_api.list_plugins().results}
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

apis = {
    'images': images_api,
    'processing': processing_api,
    'manager' : manager
}

while (True):
    jobs=processing_api.list_plugin_jobs(limit=100000000).results
    logging.info('Job queue contains '+str(len(jobs))+' jobs')
    for job in jobs:
        
        # Only work on jobs that are not already completed
        if (job.processing_complete==100):
            continue
        
        for k in plugins:
            plugin = plugins[k].plugin
            # MIDOG 2022 domain adversarial baseline
            if (job.plugin == plugins_exact[plugin['package']].id) and (job.result is None) and ((job.attached_worker is None) or (len(job.attached_worker)==0) or ((datetime.datetime.now()-job.updated_time).seconds>3600)):
                update_progress = lambda progress: processing_api.partial_update_plugin_job(id=job.id,processing_complete=progress, updated_time=datetime.datetime.now())

                apis['processing'].partial_update_plugin_job(id=job.id, attached_worker=worker_name)
                # re-check if we got the job after a random time below 1 second
                time.sleep(random.random())
                
                newjob = apis['processing'].retrieve_plugin_job(id=job.id)
                
                if (newjob.attached_worker != worker_name):
                    logging.info('There was a conflict. Worker '+newjob.attached_worker+' got the job finally')
                    continue

                logging.info('Successfully claimed job %d' % job.id)

                plugin['inference_func'](apis=apis, job=job, update_progress=update_progress)
                    
                update_progress(100.0)
                apis['processing'].partial_update_plugin_job(id=job.id, attached_worker=None)

                logging.info('Unclaimed job %d' % job.id)     
                # Break for loop to achieve refreshing of jobs list
                break          


    time.sleep(5)
    

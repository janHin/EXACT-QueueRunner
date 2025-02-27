'''entrypoint scripts'''

import logging
import click_log

logger = logging.root

FMT_STRING='%(levelname)s:%(message)s'
formatter = click_log.ColorFormatter(FMT_STRING)
handler   = click_log.ClickHandler()
handler.formatter = formatter
logger.addHandler(handler)

#STL imports
import logging
from pathlib import Path

#3rd party imports
import click

#Exact imports
from exact_sync.v1.configuration import Configuration

#local imports
from ._version import __version__
from .runner import run_loop
from .runner import ExactConnection
from .runner import PluginHandler

#TODO: Handle this somehow nicer?
from .config import username,password,serverurl
configuration = Configuration()
configuration.username = username
configuration.password = password
configuration.host     = serverurl
exact_connection = ExactConnection(configuration)

@click.group()
@click.version_option(__version__)
@click_log.simple_verbosity_option(logger,default='INFO')
def cli():
    pass

@cli.command()
@click.option("--joblimit",type=int,default=-1,help="maximum number of jobs")
@click.option("--restart",is_flag=True,help="restart upon error")
@click.option("--idlelimit",help="idle time limit in seconds",default=-1)
@click.option("--outdir",default=None,type=click.Path(exists=True,path_type=Path))
def run(joblimit:int,restart:bool,idlelimit:float,outdir:Path):
    '''Command line interface'''
    logger.info('Starting up queue_handler %s',str(__version__))

    run_loop(exact_connection,joblimit,restart,idlelimit,outdir=outdir)

@cli.command()
@click.argument("job_id",type=int)
def destroy(job_id:int):
    '''destroy job with given id'''
    logger.info('detroying job with id %d',job_id)
    exact_connection.destroy_job(job_id)

@cli.command()
@click.option("--image_name",type=str,default=None)
@click.option("--image_set",type=str,default=None)
def remove_results(image_name:str,image_set:str):
    '''remove plugin results for image(s)'''
    logger.info('queue runner remove_results')

    if image_name is None and image_set is None:
        raise click.BadOptionUsage('image_name image_set',
            'either image_name or image_set have to be specified!')

    images = exact_connection.get_images(name=image_name,image_set=image_set)

    for image in images:
        if click.confirm(f'destroy plugin result for image {image.name} with id {image.id}?'):
            exact_connection.destroy_results_for_imageid(image.id)

@cli.command()
@click.argument('job_id',type=int)
@click.argument('outdir', type=click.Path(exists=True,path_type=Path))
def upload_job_results(job_id:int,outdir:Path):
    ''''''
    logger.info('trying to salvage some results for job %d',job_id)

    plugin_handler = PluginHandler(exact_connection)

    job = exact_connection.retrieve_job(job_id)
    plugin_type = plugin_handler.get_plugin_for_job(job)

    def update_progress_func(progress:float):
        exact_connection.update_job_progress(job,progress)

    plugin_instance = plugin_type(exact_connection,outdir)
    plugin_instance.continue_inference(job)
    
@cli.command()
@click.argument('job_id',type=int)
def release_job(job_id:int):
    ''''''
    logger.info('releasing job with id %d',job_id)
    exact_connection.update_job_released(job_id)
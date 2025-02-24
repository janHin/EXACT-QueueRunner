'''entrypoint scripts'''
#STL imports
import logging

#3rd party imports
import click
from ._version import __version__

from .runner import run_loop

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.root

@click.command()
@click.version_option(__version__)
@click.option("job_limit",type=int,default=-1,help="maximum number of jobs")
@click.option("restart",is_flag=True)
@click.option("idle_limit",help="idle time limit in seconds",default=-1)
def cli(job_limit:int,restart:bool,idle_limit:float):
    '''Command line interface'''
    logger.info('Starting up queue_handler %s',str(__version__))

    run_loop(job_limit,restart,idle_limit)
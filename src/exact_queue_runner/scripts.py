'''entrypoint scripts'''
#STL imports
import logging

#3rd party imports
import click
import click_log

#local imports
from ._version import __version__
from .runner import run_loop

#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.root
click_log.basic_config(logger)


@click.command()
@click.version_option(__version__)
@click.option("--job_limit",type=int,default=-1,help="maximum number of jobs")
@click.option("--restart",is_flag=True,help="restart upon error")
@click.option("--idle_limit",help="idle time limit in seconds",default=-1)
@click_log.simple_verbosity_option(logger,default='INFO')
def cli(job_limit:int,restart:bool,idle_limit:float):
    '''Command line interface'''
    logger.info('Starting up queue_handler %s',str(__version__))

    run_loop(job_limit,restart,idle_limit)
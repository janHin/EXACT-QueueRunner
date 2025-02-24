'''entrypoint scripts'''
#STL imports
import logging

#3rd party imports
import click
import click_log

#Exact imports
from exact_sync.v1.configuration import Configuration

#local imports
from ._version import __version__
from .runner import run_loop
from .runner import ExactConnection


logger = logging.root

FMT_STRING='%(levelname)s:%(message)s'
formatter = click_log.ColorFormatter(FMT_STRING)
handler   = click_log.ClickHandler()
handler.formatter = formatter

#TODO: Handle this somehow nicer?
from .config import username,password,serverurl
configuration = Configuration()
configuration.username = username
configuration.password = password
configuration.host     = serverurl
exact_connection = ExactConnection(configuration)

@click.command()
@click.version_option(__version__)
@click_log.simple_verbosity_option(logger,default='INFO')
def cli():
    pass

@cli.command()
@click.option("--job_limit",type=int,default=-1,help="maximum number of jobs")
@click.option("--restart",is_flag=True,help="restart upon error")
@click.option("--idle_limit",help="idle time limit in seconds",default=-1)
def run(job_limit:int,restart:bool,idle_limit:float):
    '''Command line interface'''
    logger.info('Starting up queue_handler %s',str(__version__))

    run_loop(exact_connection,job_limit,restart,idle_limit)

@cli.command()
@click.argument("job",type=int)
def destroy(job_id:int):
    '''destroy job with given id'''
    exact_connection.destroy_job(job_id)

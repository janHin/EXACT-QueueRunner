#STL imports
import logging
import abc
import atexit
from pathlib import Path

#3rd party imporrts

#exact imports
from exact_sync.v1.models import (PluginResultAnnotation, PluginResult,
    PluginResultEntry, Plugin, PluginJob, Image)

#local imports
from .utils.exception import PluginExcpetion
from ..exact_connection import ExactConnection

logger = logging.getLogger(__name__)

class PluginBase(abc.ABC):
    exact_fields_dict = {}

    def __init__(self,exact_connection:ExactConnection) -> None:
        super().__init__()
        self.exact_connection = exact_connection
        self.apis = exact_connection.api_dict
        #self.update_progress_func = update_progress

        atexit.register(self._cleanup)

        self._unlinklist = []

    def update_job_progress(self,job:PluginJob,progress:float):
        self.exact_connection.update_job_progress(job,progress)

    def __getattribute__(self, name: str):
        return self.exact_fields_dict[name]

    def _cleanup(self):
        logger.info('cleaning up files')
        for path in self._unlinklist:
            if isinstance(str,path):
                path = Path(path)
            logger.debug('deleting path %s',str(path))
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
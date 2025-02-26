#STL imports
import logging

#Exact imports
from exact_sync.v1.models import PluginJob

#local imports
from .exact_connection import ExactConnection
from .plugins.registry import get_plugin_registry
from .plugins.pluginbase import PluginBase

logger = logging.getLogger(__name__)

class PluginHandler():

    def __init__(self,exact_connection:ExactConnection) -> None:
        self._local_plugins = PluginHandler.get_local_plugins()
        self._exact_plugins = exact_connection.get_exact_plugins()

    @staticmethod
    def get_local_plugins():
        '''get local plugin modules from handlers subfolder'''
        logger.info('loading plugins')
        plugins=get_plugin_registry()
        logger.debug('loaded plugins: %s',repr(plugins))
        return plugins

    def get_plugin_for_job(self,job:PluginJob)->PluginBase:
        for plugin in self._local_plugins.values():
            if job.plugin == self._exact_plugins[plugin['package']].id:
                return plugin
        raise KeyError(f'no plugin found for job {job}')
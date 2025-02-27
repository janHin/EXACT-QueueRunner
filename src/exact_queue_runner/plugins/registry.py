'''
Registry module for Exact plugin implementations
'''
#STL imports
import logging

#---Global Vars
logger = logging.getLogger(__name__)
__registered_plugins = {}

logger.debug('in registry package')

def registerplugin(exact_fields_dict:dict):
    def decorate(plugin_cls):
        plugin_cls.exact_fields_dict = exact_fields_dict
        __registered_plugins[plugin_cls.__name__] = plugin_cls
        return plugin_cls

    return decorate

def get_plugin_registry()->dict:
    #logger.debug('getting registered plugins')
    return __registered_plugins

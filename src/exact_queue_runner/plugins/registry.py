'''
Registry module for Exact plugin implementations
'''
#STL imports
import logging


from .. import plugins

#---Global Vars
logger = logging.getLogger(__name__)
__registered_plugins = {}

logger.debug('in registry package')

# def __register_plugins():
#     n_registered = 0
#     for _ , name, _ in sorted(__iter_namespace(plugins)):
#         if not name.split('.')[-1].startswith('plugin'):
#             continue
#         try:
#             logger.info('activating plugin %s',name)
#             mod = importlib.import_module(name)
#             __registered_plugins[name] = mod.plugin
#         except Exception as e:
#             raise RuntimeError('+++ Unable to activate plugin: '+name) from e
#         n_registered += 1
#     if n_registered <= 0:
#         raise RuntimeError('registered no plugins')

# __register_plugins()

def registerplugin(exact_fields_dict:dict):
    def decorate(plugin_cls):
        plugin_cls.exact_fields_dict = exact_fields_dict
        __registered_plugins[plugin_cls.__name__] = plugin_cls
        return plugin_cls

    return decorate

def get_plugin_registry()->dict:
    #logger.debug('getting registered plugins')
    return __registered_plugins

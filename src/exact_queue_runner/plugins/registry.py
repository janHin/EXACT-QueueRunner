'''
Registry module for Exact plugin implementations
'''
#STL imports
import logging
import pkgutil
import importlib

from .. import plugins

#---Global Vars
logger = logging.getLogger(__name__)
__registered_plugins = {}

logger.debug('in registry package')

def __iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

def __register_plugins():
    logger.info('registering plugins')
    raise RuntimeError()
    for _ , name, _ in sorted(__iter_namespace(plugins)):
        if not name.startswith('plugin'):
            continue
        try:
            logger.info('activating plugin %s',name)
            mod = importlib.import_module(name)
            __registered_plugins[name] = mod.plugin
        except Exception as e:
            raise RuntimeError('+++ Unable to activate plugin: '+name) from e

__register_plugins()

def get_plugin_registry()->dict:
    #logger.debug('getting registered plugins')
    return __registered_plugins

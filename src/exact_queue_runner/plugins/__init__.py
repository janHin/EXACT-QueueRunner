#STL imports
import pkgutil
import importlib
import logging

#local imports
from .. import plugins

logger = logging.getLogger(__name__)

# def __iter_namespace(ns_pkg):
#     # Specifying the second argument (prefix) to iter_modules makes the
#     # returned name an absolute name instead of a relative one. This allows
#     # import_module to work without having to do additional modification to
#     # the name.
#     return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

def __load_local_modules():

    n_registered = 0
    for _ , name, _ in sorted(pkgutil.iter_modules(__path__,__name__ + ".")):
        module_name = name.split('.')[-1]
        if not module_name.startswith('plugin'):
            continue
        try:
            logger.info('activating plugin %s',name)
            mod = importlib.import_module(name)
        except Exception as e:
            raise RuntimeError('+++ Unable to activate plugin: '+name) from e
        n_registered += 1
    if n_registered <= 0:
        raise RuntimeError('registered no plugins')

__load_local_modules()
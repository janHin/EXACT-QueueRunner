#STL imports
import pkgutil
import socket
import string
import random

def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

def get_workername()->str:
    ''''''
    workername = str(socket.gethostname()+'_'+''.join(random.choice(string.ascii_uppercase +
                                                    string.ascii_lowercase +
                                                    string.digits)
                                    for _ in range(6)))
    return workername
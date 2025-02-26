#STL imports
import pkgutil
import socket
import string
import random


def get_workername()->str:
    ''''''
    workername = str(socket.gethostname()+'_'+
                     ''.join(random.choice(string.ascii_uppercase +
                                           string.ascii_lowercase +
                                           string.digits)
                            for _ in range(6))
                    )
    return workername
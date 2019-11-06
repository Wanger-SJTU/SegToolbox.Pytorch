from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# subclass dict and define getter-setter. 
# This behaves as both dict and obj
class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            else:
                self.__dict__[key] = value
        else:
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            else:
                self[key] = value
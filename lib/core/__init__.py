# -*- coding: UTF-8 -*-
# __init__.py
# @author wanger
# @description 
# @created 2019-10-28T15:54:57.838Z+08:00
# @last-modified 2019-10-29T10:00:17.200Z+08:00
#

from .config import cfg_from_file
from .config import cfg_from_list
from .config import config

__all__ = ["cfg_from_file", "cfg_from_list","config"]
#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/7 10:34 
# @Author : yangpingyan@gmail.com
import os
from log import log


def get_workdir():
    try:
        workdir = os.path.abspath(__path__)
    except:
        workdir = os.getcwd()
    return workdir

workdir = get_workdir()

# configure debug_mode automatically just for convenience
debug_mode = False
if workdir.find('iCloud') != -1:
    debug_mode = True
log.debug("debug_mode is {}".format(debug_mode))



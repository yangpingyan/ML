#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/7 10:34 
# @Author : yangpingyan@gmail.com
import os
debug_mode = True

# configure debug_mode automatically just for convenience



def get_workdir():
    try:
        workdir = os.path.abspath(__path__)
    except:
        workdir = os.getcwd()
    return workdir

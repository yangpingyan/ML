#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/7 10:34 
# @Author : yangpingyan@gmail.com
import os

def get_workdir(projectid):
    try:
        workdir = os.path.abspath(__path__)
    except:
        workdir = os.getcwd()
        if workdir.find(projectid) == -1:
            workdir = os.path.join(workdir, projectid)
    return workdir

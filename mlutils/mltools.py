#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/7 10:34 
# @Author : yangpingyan@gmail.com
import os

def get_workdir(projectid):
    try:
        cur_dir = os.path.dirname(__file__)
    except:
        cur_dir = os.getcwd()
    if cur_dir.find(projectid) == -1:
        os.chdir(projectid)
    return cur_dir
#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/7 10:34 
# @Author : yangpingyan@gmail.com
import os
from mibao_log import log

def get_csv_files(dir_path):
    '''获取dir_path当前目录下的所有csv文件（不包含子目录）'''
    L = []
    #    for root, dirs, files in os.walk(file_dir):  #搜索子目录
    files = os.listdir(dir_path)
    for file in files:
        if os.path.splitext(file)[1] == '.csv' \
            and file.startswith('.') is False \
            and file.startswith('~') is False:
            L.append(os.path.join(dir_path, file))
    return L

def get_workdir():
    try:
        workdir = os.path.dirname(os.path.realpath(__file__))
    except:
        workdir = os.getcwd()
    return workdir

workdir = get_workdir()

# configure debug_mode automatically just for convenience
debug_mode = False
if workdir.find('github') != -1:
    debug_mode = True
log.debug("debug_mode is {}".format(debug_mode))



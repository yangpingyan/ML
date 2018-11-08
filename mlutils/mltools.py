#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/7 10:34 
# @Author : yangpingyan@gmail.com
import os


def get_workdir(projectid):
    cur_dir = os.getcwd()
    print(cur_dir)
    if cur_dir.find(projectid) == -1:
        cur_dir = os.path.join(cur_dir, projectid)
    return cur_dir

#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/9/28 18:07 
# @Author : yangpingyan@gmail.com

import time
from mlutils import *
import pandas as pd

start_time = time.clock()
df = load_data_mibao()
df.to_csv("d:/order_user.csv", index=False)
print(time.clock()-start_time)

# df = pd.read_csv("d:/order_user.csv")
# df.columns.values
# df['cert_no'].value_counts()
# missing_values_table(df)
# print(df)

print("Mission Complete")
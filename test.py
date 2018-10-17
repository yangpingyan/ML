#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/10/17 12:06 
# @Author : yangpingyan@gmail.com
import json
import pandas as pd
import numpy as np


tmp = pd.read_csv(datasets_path + "order_phone_book.csv")
indexId = tmp[tmp['order_id'] == 59614].index.tolist()[0]
phoneBook = tmp.at[indexId, 'phone_book']
tmp['phone_book'].dtype
tmp[tmp['order_id'] == 61234]['phone_book'].str.count("\"name\":\"\"")
pp['name']
len(pp)
type(pp)
print(phoneBook)
df.sort_values(by='phone_book', inplace=True)


def count_name_nums(data):
    data_list = json.loads(data)
    name_list = []
    for phone_book in data_list:
        if len(phone_book.get('name')) > 0 and phone_book.get('name').isdigit() is False:
            name_list.append(phone_book.get('name'))

     return len(set(name_list))
cc = count_name_nums(phoneBook)
print(cc)

df = all_data_df
df[['card_id', 'cert_no']][df['card_id'] != df['cert_no']]

df['delivery_way'].value_counts()
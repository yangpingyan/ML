#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/9/28 16:42 
# @Author : yangpingyan@gmail.com

# 导入必要模块
import pandas as pd
from sqlalchemy import create_engine

# 初始化数据库连接，使用pymysql模块
# MySQL的用户：root, 密码:147369, 端口：3306,数据库：mydb
engine = create_engine('mysql+pymysql://root:147369@localhost:3306/mydb')

# 查询语句，选出employee表中的所有数据
sql = '''
      select * from employee;
      '''

# read_sql_query的两个参数: sql语句， 数据库连接
df = pd.read_sql_query(sql, engine)

# 输出employee表的查询结果
print(df)

# 新建pandas中的DataFrame, 只有id,num两列
df = pd.DataFrame({'id':[1,2,3,4],'num':[12,34,56,89]})

# 将新建的DataFrame储存为MySQL中的数据表，不储存index列
df.to_sql('mydf', engine, index= False)

print('Read from and write to Mysql table successfully!')

作者：但盼风雨来_jc
链接：https://www.jianshu.com/p/238a13995b2b
來源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
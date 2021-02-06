# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:32:08 2021

@author: Wountry
"""


import pandas as pd
import os


output='D:\\Pythondata\\qcr.csv'
input_path='D:/Pythondata/'
files=os.listdir(input_path)
all_data=[]
for files in files:
    data_frame=pd.read_csv(input_path+files,index_col=0)
    all_data.append(data_frame)
data= pd.concat(all_data, axis=0, ignore_index=True)#拼接数据
data=data.dropna(axis=0,how = 'all')  #删除空行
data = data.reset_index(drop=False)#添加索引
data.to_csv(output, index=False)

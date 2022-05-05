#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 21:13:06 2022

@author: user
"""

# Huge thanks to Shauna!!!!

data= # dictionary here

num_rows = len(data['data'])

import pandas as pd


data_list = []
for i in range(len(data['data'])):
    #print(i)
    data_list.append([data['data'][i]['id'], data['data'][i]['text'].encode('UTF-8','ignore').decode('UTF-8')])
    
    
    
data_df = pd.DataFrame(data_list)
data_df.columns= [['id','text']]

data_df.to_csv(r"D:\CU\spring 2022\nlp\Roku\sep_oct.csv")#, encoding = "utf-8")


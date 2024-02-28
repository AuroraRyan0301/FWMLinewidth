import pandas as pd
import numpy as np
import os
import sys
from utils import *

# 读取Cd59219.csv文件
df = pd.read_csv('Cd53310.csv', encoding='utf-8')

# 创建一个字典，将能级波数和label对应
# labeldict = {"51483":"2e:0", 
#              "53310":"0e:1",
#              "59485":"2e:1",
#              "59497":"4e:1",
#              "62563":"2e:2",
#              "63086":"0e:2",
#              "65134":"4e:2",
#              "65353":"2e:3",
#              "65358":"4e:3",}
labeldict = {"53310":"0e:1",}

# 读取Cd new.csv文件
df2 = pd.read_csv('Cd_gleb.csv', encoding='utf-8')

item_count = 20
# 根据labeldict字典，在 Unnamed: 2 列查找每个key（能级）对应的value（label），如果相同提取Unname: 1列，写入df的fb列
# iterate over the rows of the DataFrame
for i, (key, value) in enumerate(labeldict.items()):
    count = 0
    count2 = 0
    newdatacol = pd.DataFrame(columns=["New_fb","New_gf"])
    for index, row in df2.iterrows():

        if str(row["Unnamed: 2"]).strip() == "0e:0" and count < item_count:
            print("label {},count {}".format("0e:0", count))
            # if there is a match, extract the value from the "Unnamed: 1" column and write it to the "fb" column
            newdatacol.loc[count, "New_gf"] = row["Unnamed: 1"]
            count += 1
        # check if the value in the "Unnamed: 2" column matches a key in "labeldict"
        if str(row["Unnamed: 2"]).strip() == value:
            print("label {},count {}".format(value, count2))
            # if there is a match, extract the value from the "Unnamed: 1" column and write it to the "fb" column
            newdatacol.loc[count2, "New_fb"] = row["Unnamed: 1"]
            count2 += 1
            if count2 == item_count:
                break
    # import ipdb; ipdb.set_trace()
    df["fb"] = newdatacol["New_fb"]
    df["gf"] = newdatacol["New_gf"]
    # 保存为新文件 Cd{}.csv .format(key)
    newpath = "Cd{}_new.csv".format(key)
    df.to_csv(newpath, index=False)
    csv_wavnum2freq(newpath)
    csv_mu2f(newpath,jg=0,jf=1)





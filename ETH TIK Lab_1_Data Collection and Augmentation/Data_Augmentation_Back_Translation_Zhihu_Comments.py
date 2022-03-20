"""
Author: Runyao Yu
runyao.yu@tum.de
Research Internship in ETH Zurich
For Academic Use Purpose only
"""

import pandas as pd
import jionlp as jio
import random
import numpy as np

def group_data(content,cla):
  cla = [int(x) for x in cla] #将浮点数变成整数才能分类/transform to int
  temp_cla = [] 
  temp_content = [] #comments文本/comments texts
  temp_content_p = [] #正例的类/positive comments class
  for i in range(len(cla)):
    if cla[i] == 1:
      temp_content.append(content[i]) 
      temp_content_p.append(content[i]) #正例的文本/positive comments
      temp_cla.append(cla[i]) 

  len_temp_cla_p = len(temp_cla) #正例的类的长度/positive comments length   

  for j in range(len(cla)):
    if cla[j] == 0:
      temp_content.append(content[j])
      temp_cla.append(cla[j])

  return temp_content, temp_cla, len_temp_cla_p, temp_content_p #分别返回 排序后的comments，标签，正例的类的长度， 正例的类/data after resorting


def back_translate(back_trans, data, TEXT, len_temp_cla_p, temp_content_p, name):
  len_temp_cla_n = len(TEXT) - len_temp_cla_p #负例的长度/negative comments length   
  delta = len_temp_cla_n - len_temp_cla_p #需要新生成的正例的个数/how many new positive comments that need to be generated
  times = int(delta/5) #每次可以生成5个，所以只需要运行delta/5次/each time can generate 2-7 new data

  if times < len_temp_cla_p:
    temp_list = []
    for i1 in range(times):
      temp_list.append(random.choice(temp_content_p)) #如果需要运行的次数比正例总数小，直接赋值/logic for running times
  else:
    temp_list = []
    for i2 in range(len_temp_cla_p):
      temp_list.append(random.choice(temp_content_p))
  len_list = len(temp_list)
  translated = []
  new_label = []
  for m in range(len(temp_list)):
    Trans = back_trans(temp_list[m])
    for n in range(len(Trans)): #每次翻译生成N个文本，for循环添加N个/each time add many augmented data
      translated.append(Trans[n])
      new_label.append(int(1))
  print(Trans)
  new_data = {'text':translated, 'label':new_label}
  new_data = pd.DataFrame(new_data)
  new_df = pd.DataFrame(np.concatenate([data.values, new_data.values]), columns=data.columns) #将新生成的data和原本的data连接起来/concate raw data and augmentated data
  new_df.to_excel('Augmentated_{}'.format(name)+'.xlsx', index = False, header=True)

#修改第一行的路径/change the path
df = pd.read_excel("/Raw_Data_Sep20.xlsx")
df = df.fillna(0)

temp_df = df.drop('content', axis=1)
classes = []
for col_name in temp_df.columns: 
  classes.append(col_name) #得到各个列的名字存进list/store column's name in loop

#修改api和密码/change the appid and secretKey, you can register in this wed: http://api.fanyi.baidu.com/api/trans/product/index , select "注册通用翻译"
baidu_api = jio.BaiduApi([{'appid': '20210920000950997', 'secretKey':'wDfYZ7RglkhBe1yXapL0'}], gap_time=0.3) 
apis = [baidu_api]  #可根据需要进行扩展其他翻译器/you can add other translators
back_trans = jio.BackTranslation(mt_apis=apis)

for name in classes:
  cla = df['{}'.format(name)].tolist() #遍历每一个类， 把标签储存为list/go through all the classes
  content = df['content'].tolist()  #comments文本/comments texts
  temp_content, temp_cla, len_temp_cla_p, temp_content_p = group_data(content,cla) #分别返回 排序后的comments，标签，正例的类的长度， 正例的类/use function above
  data = {'text': temp_content, '{}'.format(name): temp_cla}  
  data = pd.DataFrame(data)
  data.to_excel('{}'.format(name)+'.xlsx', index = False, header=True) #排过序的raw data/raw data after resorting
  back_translate(back_trans, data, temp_content, len_temp_cla_p, temp_content_p, name) 
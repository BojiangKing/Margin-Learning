#%%
from collections import defaultdict
import os
import hanlp
from hanlp.components.mtl.multi_task_learning import MultiTaskLearning
from hanlp_common.document import Document
from pandas.core.frame import DataFrame
HanLP: MultiTaskLearning = hanlp.load(hanlp.pretrained.mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
import pandas as pd
import numpy as np
from PIL import Image  # 处理图片
from matplotlib import pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
#%%
def pick_tasks(pick_tasks):
    tasks = list(HanLP.tasks.keys())
    # Pick what you need from what we have
    for task in tasks:
        if task not in pick_tasks:
            del HanLP[task]
    # You can save it as a new component
    # HanLP.save(path)
    # print(HanLP.tasks.keys())
#%%
def draw_word_cloud(result, mask_path=None):
    # 生成对应的词云

    if mask_path and os.path.exists(mask_path):
        # 利用PIL的Image打开背景图片，并且进行相应的参数化
        img = Image.open(mask_path)

        graph = np.array(img)  # 把img的参数给到graph生成相应的词云
        # 进一步，基于背景颜色，设置字体的颜色
        image_color = ImageColorGenerator(graph)
    else:
        graph = None

    # WordCloud默认不支持中文，加载对应的中文黑色字体库，一般在电脑的C盘  fonts

    # mask以传递过来的数据绘制词云
    wc = WordCloud(font_path='‪C:\Windows\Fonts\simhei.ttf', background_color='white', max_words=300, mask=graph)

    # 将字典的文本生成相对应的词云
    wc.generate_from_frequencies(result)
    # wc.fit_words(dic_word_cloud)

    # 显示对应的图片
    plt.imshow(wc)  # 显示对应的词云图
    plt.axis('off')  # 关闭图像对应的坐标

    plt.show()  # 显示对应的窗口

    # 将生成的图片保存
    # wc.to_file('H:\大数据分析\携程舆情分析\酒店差评词云.jpg')
#%%
def load_file(filename):
    with open(filename,'r',encoding="utf-8") as f:
        contents = f.readlines()
    result = []
    for content in contents:
        result.append(content.strip())
        
    return result
#%%
def remove_stopwords(segs):
    result = []
    for s in segs:
        if s not in stopwords:
            result.append(s)
    return result
        
# %%
file_path = r'./Files/TMSJ2.xls'
df = pd.read_excel(file_path, sheet_name = "Sheet1")
stopwords = load_file('./Files/stopwords.txt')
pick_tasks(['tok','ner', 'dep'])
# %%
print(HanLP.tasks.keys())
entity_dic = defaultdict(int)
for i in range(len(df)):
    doc: Document = HanLP(df.iloc[i,1:3].tolist())
    tokens = remove_stopwords(doc['tok'])
    for list_items in tokens:
        for item in list_items:
            entity_dic[item] += 1
# doc.pretty_print()
entity_dic
# %%
# print(HanLP.tasks.keys())
from pyhanlp import HanLP as hp

entity_dic = defaultdict(int)
for i in range(len(df)):
    doc: Document = hp.extractKeyword(df.iloc[i,1]+df.iloc[i,2], 20)
    tokens = remove_stopwords(doc)
    for token in tokens:
        entity_dic[token] += 1
# entity_dic
#%%
draw_word_cloud(entity_dic)
# %%
doc: Document = HanLP(df.iloc[0,1:3].tolist())
doc
# %%
doc.pretty_print()
# %%
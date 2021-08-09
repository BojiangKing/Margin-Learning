#%%
from typing import DefaultDict
import pandas as pd
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
#%%
import re
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")
#%%
file_path = r'./Files/TMSJ2.xls'
df = pd.read_excel(file_path, sheet_name = "Sheet1") # sheet_name不指定时默认返回全表数据
tr4w = TextRank4Keyword()
map_keys = DefaultDict(int)
map_phrases = DefaultDict(int)
len_df = len(df)
# %%
for i in range(len_df):
    text = df.iloc[i,1] + '\n' + df.iloc[i,2]
    tr4w.analyze(text=text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    for item in tr4w.get_keywords(5, word_min_len=1):
        map_keys[item.word] += 1
    for phrase in tr4w.get_keyphrases(keywords_num=2, min_occur_num= 2):
        map_phrases[phrase] += 1
# print(map_keys)
dict= sorted(map_keys.items(), key=lambda d:d[1], reverse = True)
with open(r'./Files/Summary_keywords.txt', 'w', encoding='utf-8') as f:
    for items in dict:
        f.write(items[0] + '  ' + str(items[1]) + '\n')
dict= sorted(map_phrases.items(), key=lambda d:d[1], reverse = True)
with open(r'./Files/Summary_phrases.txt', 'w', encoding='utf-8') as f:
    for items in dict:
        f.write(items[0] + '  ' + str(items[1]) + '\n')
# %%
lst_sentences = []
tr4s = TextRank4Sentence()
for i in range(len_df):
    lst_sentences.append(df.iloc[i,1])
    tr4s.analyze(text=df.iloc[i,2], lower=True, source = 'all_filters')
    num = len(cut_sent(df.iloc[i,2])) // 5
    for item in tr4s.get_key_sentences(num=num):
        lst_sentences.append(item.sentence)
    lst_sentences.append("")
with open(r'./Files/Summary_abstracts.txt', 'w', encoding='utf-8') as f:
    for item in lst_sentences:
        f.write(item + '\n')
# %%
import complex_sentence
handler = complex_sentence.TextMining()
result = handler.process_mongonews(lst_sentences)
# %%
result
# %%
from fastHan import FastHan
model=FastHan(model_type='large',url="./finetuned_model")

#%%
def cut_partion(para):
    splitwords = ['方案(建议解决办法)：','方案（建议解决办法）：','建议解决办法：','为此建议：']
    for s in splitwords:
        if s in para:
            return para.split(s)
# %%
map_keys_w = DefaultDict(int)
map_phrases_w = DefaultDict(int)
map_keys_j = DefaultDict(int)
map_phrases_j = DefaultDict(int)
for i in range(len_df):
    partaions = cut_partion(df.iloc[i,2])
    if len(partaions) != 2 or len(partaions[0]) <= 0 or len(partaions[0]) <= 0:
        continue
    tr4w.analyze(text=partaions[0], lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    for item in tr4w.get_keywords(5, word_min_len=1):
        map_keys_w[item.word] += 1
    for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num= 2):
        map_phrases_w[phrase] += 1

    tr4w.analyze(text=partaions[1], lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    for item in tr4w.get_keywords(5, word_min_len=1):
        map_keys_j[item.word] += 1
    for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num= 2):
        map_phrases_j[phrase] += 1
# print(map_keys)
dict= sorted(map_keys_w.items(), key=lambda d:d[1], reverse = True)
with open(r'./Files/Summary_keywords_w.txt', 'w', encoding='utf-8') as f:
    for items in dict:
        f.write(items[0] + '  ' + str(items[1]) + '\n')
dict= sorted(map_keys_j.items(), key=lambda d:d[1], reverse = True)
with open(r'./Files/Summary_keywords_j.txt', 'w', encoding='utf-8') as f:
    for items in dict:
        f.write(items[0] + '  ' + str(items[1]) + '\n')
dict= sorted(map_phrases_w.items(), key=lambda d:d[1], reverse = True)
with open(r'./Files/Summary_phrases_w.txt', 'w', encoding='utf-8') as f:
    for items in dict:
        f.write(items[0] + '  ' + str(items[1]) + '\n')
dict= sorted(map_phrases_j.items(), key=lambda d:d[1], reverse = True)
with open(r'./Files/Summary_phrases_j.txt', 'w', encoding='utf-8') as f:
    for items in dict:
        f.write(items[0] + '  ' + str(items[1]) + '\n')
# %%
lst_sentences = []
wordnums = 5
tr4w = TextRank4Keyword()
tr4s = TextRank4Sentence()
for i in range(len_df):
    lst_sentences.append(df.iloc[i,1])
    partaions = cut_partion(df.iloc[i,2])
    if len(partaions) != 2 or len(partaions[0]) <= 0 or len(partaions[0]) <= 0:
        continue
    tr4w.analyze(text=partaions[0], lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    w_k = '问题关键词：'
    for item in tr4w.get_keywords(wordnums, word_min_len=1):
        w_k += (item.word + ' ')
    w_p = '问题关键短语：'
    for phrase in tr4w.get_keyphrases(keywords_num=2, min_occur_num= 2):
        w_p += (phrase + ' ')
    lst_sentences.append(w_k)
    lst_sentences.append(w_p)
    lst_sentences.append('问题摘要')
    tr4s.analyze(text=partaions[0], lower=True, source = 'all_filters')
    num = len(cut_sent(partaions[0])) // 3 + 1
    for item in tr4s.get_key_sentences(num=num):
        lst_sentences.append(item.sentence)

    tr4w.analyze(text=partaions[1], lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    j_k = '建议关键词：'
    for item in tr4w.get_keywords(wordnums, word_min_len=1):
        j_k += (item.word + ' ')
    j_p = '建议关键短语：'
    for phrase in tr4w.get_keyphrases(keywords_num=2, min_occur_num= 2):
        j_p += (phrase + ' ')
    lst_sentences.append(j_k)
    lst_sentences.append(j_p)
    lst_sentences.append('建议摘要')
    tr4s.analyze(text=partaions[1], lower=True, source = 'all_filters')
    num = len(cut_sent(partaions[1])) // 3 + 1
    for item in tr4s.get_key_sentences(num=num):
        lst_sentences.append(item.sentence)
    lst_sentences.append("")
with open(r'./Files/Summary_abstracts.txt', 'w', encoding='utf-8') as f:
    for item in lst_sentences:
        f.write(item + '\n')
# %%
import numpy as np
from PIL import Image  # 处理图片
from matplotlib import pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

dict= sorted(map_keys.items(), key=lambda d:d[1], reverse = True)
dic_word_cloud = {}
for i in range(10):
    dic_word_cloud[dict[i][0]] = dict[i][1] / 100

# 生成对应的词云

# 利用PIL的Image打开背景图片，并且进行相应的参数化
img = Image.open('image.png')

graph = np.array(img)  # 把img的参数给到graph生成相应的词云

# WordCloud默认不支持中文，加载对应的中文黑色字体库，一般在电脑的C盘  fonts

# mask以传递过来的数据绘制词云
wc = WordCloud(font_path='‪C:\Windows\Fonts\simhei.ttf', background_color='white', max_words=300)

# 将字典的文本生成相对应的词云
wc.generate_from_frequencies(dic_word_cloud)
# wc.fit_words(dic_word_cloud)
# 进一步，基于背景颜色，设置字体的颜色
image_color = ImageColorGenerator(graph)

# 显示对应的图片
plt.imshow(wc)  # 显示对应的词云图
plt.axis('off')  # 关闭图像对应的坐标

plt.show()  # 显示对应的窗口

# 将生成的图片保存
# wc.to_file('H:\大数据分析\携程舆情分析\酒店差评词云.jpg')
# %%

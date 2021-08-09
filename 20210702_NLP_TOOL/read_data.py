#%%
import pandas as pd
#%%
file_path = r'./Files/TMSJ2.xls'
df = pd.read_excel(file_path, sheet_name = "Sheet1") # sheet_name不指定时默认返回全表数据
# %%
with open(r'./Files/TMSJ2-total.txt', 'w', encoding='utf-8') as f:
    for i in range(len(df)):
        f.write(df.iloc[i,1] + '\n' + df.iloc[i,2] + '\n')
# %%
from fastHan import FastHan
model=FastHan('large', url='./finetuned_model')
# %%
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
for i in range(len(df)):
    sentences = cut_sent(df.iloc[i,1])
    sentences.extend(cut_sent(df.iloc[i,2]))
    answer=model(sentences,target="Parsing")
    for a in answer:
        print(a)


#%%
# traindata file path
cws_url=r'./Files/TMSJ2-total-53.txt.anns'

model.set_device('cpu')

model.finetune(data_path=cws_url,task='NER',save=True,save_url='finetuned_model')
# %%
from fastHan import FastHan
model=FastHan(model_type='large')
cws_url=r'./Files/TMSJ2-total-53.txt.anns'
model.train_test(cws_url)

# %%

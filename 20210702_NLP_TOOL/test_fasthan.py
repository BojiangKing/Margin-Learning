from fastHan import FastHan
import pandas as pd
model=FastHan('large', url='./finetuned_model')

# sentence="郭靖是金庸笔下的男主角。"
# answer=model(sentence)
# print(answer)
# answer=model(sentence,target="Parsing")
# print(answer)
# answer=model(sentence,target="NER")
# print(answer)

import re

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


from nltk import DependencyGraph
def drawtree(answer):
    
    for words in answer:
        par_result = ''
        for word in words:
            if word[1] == 0:
                word[2] = "ROOT"
            par_result += "\t" + word[0] + "(" + word[2] + ")" + "\t" + word[3] + "\t" + str(word[1]) + "\t" + word[2] + "\n"
        conlltree = DependencyGraph(par_result)  # 转换为依存句法图
        tree = conlltree.tree()  # 构建树结构
        tree.draw()  # 显示输出的树





# sentences = cut_sent('建立以发改、商务、规划等相关职能部门为牵头单位，属地镇街为配合单位的管理体系，成立楼宇发展办公室，统一编制发展规划，统一划分楼宇区块，统一整合配套资源，统一出台扶持政策，并将楼宇经济纳入年度岗位目标责任制考核体系，建立以行政审批中心为牵头单位，工商、税务、人力社保、科技、公安等职能部门为融合单位的服务体系，设立楼宇大数据信息中心及楼宇企业服务中心、科技创新发展中心，功能辐射*、华舍两大楼宇企业服务中心')
# # answer=model(sentence)
# # print(answer)
# answerParsing=model(sentences,target="Parsing")
# # print(answerParsing)
# # answerNER=model(sentences,target="NER")
# # print(answer)
# drawtree(answerParsing)

file_path = r'./Files/TMSJ2.xls'
df = pd.read_excel(file_path, sheet_name = "Sheet1") 
for i in range(len(df)):
    sentences = cut_sent(df.iloc[i,1])
    sentences.extend(cut_sent(df.iloc[i,2]))
    answer=model(sentences,target="Parsing")
    drawtree(answer)



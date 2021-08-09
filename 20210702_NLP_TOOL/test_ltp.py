import sys, os
from pyltp import *
 
from nltk import DependencyGraph
 
 
class LtpParsing(object):
    def __init__(self, model_dir='ltp_data'):
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(model_dir, "cws.model"))
        self.postagger = Postagger()
        self.postagger.load(os.path.join(model_dir, "pos.model"))
        self.parser = Parser()
        self.parser.load(os.path.join(model_dir, "parser.model"))
 
    def par(self, infilm, outfilm):
        input_data = open(infilm, 'r', encoding='utf-8')
        output_data = open(outfilm, 'w+', encoding='utf=8')
        for line in input_data.readlines():
            line = line.strip()
            # 分词
            words = self.segmentor.segment(line)
            # self.segmentor.load_with_lexicon('lexicon')  # 使用自定义词典，lexicon外部词典文件路径
            # print('分词：' + '\t'.join(words))
 
            # 词性标注
            postags = self.postagger.postag(words)
            # print('词性标注：' + '\t'.join(postags))
 
            # 句法分析
            arcs = self.parser.parse(words, postags)
            rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
            relation = [arc.relation for arc in arcs]  # 提取依存关系
            heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
 
            output_data.write(line)
            output_data.write('\n')
            output_data.write('句法分析：')
            par_result = ''
            for i in range(len(words)):
                if arcs[i].head == 0:
                    arcs[i].relation = "ROOT"
                par_result += "\t" + words[i] + "(" + arcs[i].relation + ")" + "\t" + postags[i] + "\t" + str(arcs[i].head) + "\t" + arcs[i].relation + "\n"
                output_data.write(relation[i] + '(' + words[i] + ', ' + heads[i] + ')' + '\n')
            # print(par_result)
            conlltree = DependencyGraph(par_result)  # 转换为依存句法图
            tree = conlltree.tree()  # 构建树结构
            tree.draw()  # 显示输出的树
            output_data.write('\n')
        input_data.close()
        output_data.close()
 
    def release_model(self):
        # 释放模型
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()
 
if __name__ == '__main__':
    infilm = 'infilm.txt'
    outfilm = 'outfilm.txt'
    ltp = LtpParsing()
    ltp.par(infilm, outfilm)
    ltp.release_model()
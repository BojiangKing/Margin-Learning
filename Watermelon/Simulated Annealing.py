#%%
'''使用SA算法求解TSP问题'''
import numpy as np
import time
import random
import matplotlib.pyplot as plt

T_start = 5000.0#初始温度
T_end = 1e-8#终止温度
q_ratio = 0.98#温度衰减系数
L_max = 1000#同一个温度下迭代次数
Num_of_city = 31#城市总数
#城市坐标
cityList = [(1304,2312),(3639,1315),(4177,2244),(3712,1399),
            (3488,1535),(3326,1556),(3238,1229),(4196,1004),
            (4312,790),(4386,570),(3007,1970),(2562,1756),
            (2788,1491),(2381,1676),(1332,695),
            (3715,1678),(3918,2179),(4061,2370),
            (3780,2212),(3676,2578),(4029,2838),
            (4263,2931),(3429,1908),(3507,2367),
            (3394,2643),(3439,3201),(2935,3240),
            (3140,3550),(2545,2357),(2778,2826),
            (2370,2975)
]

def create_new(planing):
    '''输入当前的方案，随即交换两个位置的城市，产生一个新的临域方案'''
    pos1 = random.randint(0,Num_of_city-1)
    pos2 = random.randint(0,Num_of_city-1)
    planing[pos2],planing[pos1] = planing[pos1],planing[pos2]
    return planing

def distance(startplaceindex,endplaceindex):
    '''计算两座城市之间的距离，输入为城市索引'''
    x1 = cityList[startplaceindex][0]
    y1 = cityList[startplaceindex][1]
    x2 = cityList[endplaceindex][0]
    y2 = cityList[endplaceindex][1]
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def path_len(planing):
    '''计算当前方案下的总距离'''
    path = 0 #初始化总距离
    #计算第一到第三十一个城市的总距离
    for i in range(Num_of_city-1):
        dis = distance(planing[i],planing[i+1])
        path +=dis
    #累加计算从第三十一个城市回到第一个城市的距离
    last_dist = distance(planing[-1],planing[0])
    return path+last_dist

if __name__ == "__main__":
    
    np.random.seed(42)
    T = T_start#当前温度
    best_planing = []#最好的方案
    # best_cost = float(inf)#最优方案下的距离
    count = 0#降温迭代次数

    start_time = time.time()
    #构造初始解
    # current_planing = np.random.permutation(Num_of_city)#用随机的方式构造初始解
    current_planing =  [num for num in range(Num_of_city)]#直接从0-1-2-3-……-30-0（从0开始）

    #模拟退火搜索过程
    while(T>=T_end):
        for i in range(L_max):
            #每一个温度下要迭代L_max次
            best_planing = current_planing.copy()
            current_planing = create_new(current_planing)
            f1 = path_len(best_planing)#计算原来方案的总距离
            f2 = path_len(current_planing)#计算新方案的总距离
            df = f2 - f1 #优化量
            if df >= 0:
                r_prod = random.random()
                if r_prod >= np.exp(-df/T):
                    #随机生成概率比阈值大，拒绝新解,恢复原来的解
                    current_planing = best_planing.copy()
        T *= q_ratio
        count += 1

    end_time = time.time()
    duration = end_time - start_time
    print("模拟退火算法，初始温度T0={},降温系数q={},每个温度迭代{}次,共降温{}次，得到的TSP最优路径为:\n".format(T_start,q_ratio,L_max,count))
    for i in range(Num_of_city):
        print(best_planing[i],'-->',end='')
    print(best_planing[0])
    print("最优路径长度：",path_len(best_planing))
    print("程序运行时间：",duration)
    #绘制行驶路线图
    best_planing.append(best_planing[0])
    x = [cityList[index][0] for index in best_planing]
    y = [cityList[index][1] for index in best_planing]
    plt.figure(figsize=(15,10))
    plt.plot(x,y,'o')
    plt.plot(x,y,linewidth=1,color='red')
    plt.plot(x[0],y[0],'v',markersize=20)
    plt.title('SA_TSP')
    plt.show()
# %%

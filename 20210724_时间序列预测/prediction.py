#%%
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
# %%
def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    return model.fit(disp=0)
#%%
basepath = r'D:\King\文件\2021-06\20210610_碳中和\Data\CEADS'
filename = 'Zhejiang1997~2018.xlsx'
industry_names = []
columns = ["Year"]
for i in range(1997,2019):
    sheetname=str(i)
    data = pd.read_excel(fr'{basepath}\{filename}', sheet_name=sheetname, header=0)
    if i == 1997:
        industry_names = data.iloc[2:50,0].tolist()
        columns.extend(industry_names)
        datas = pd.DataFrame(columns=columns)
    subdata = [f'{i}-12-31']
    subdata.extend(data.loc[2:50,"Total"])
    datas.loc[i-1997] = subdata
#%%
timeseries = datas.copy()
timeseries["Year"] = pd.to_datetime(timeseries['Year'])
timeseries = timeseries.set_index('Year').asfreq('Y')
# decompose = seasonal_decompose(datas, model='ad')
# decompose

# %%
lags = 3
# trend = decompose.trend
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# sm.graphics.tsa.plot_acf(trend, lags=lags, ax=ax1)
# ax2 = fig.add_subplot(212)
# sm.graphics.tsa.plot_pacf(trend, lags=lags, ax=ax2)
# plt.show()
trend_evaluates = []
map_pq = {}
for key in range(48):
    print("============================================" + str(key))
    indexs=[i for i,x in enumerate(timeseries.iloc[:,key]) if timeseries.iat[i,key]!=0]
    if len(indexs) >= lags:
        timeserie = timeseries.iloc[indexs,key]
        trend_evaluate = sm.tsa.arma_order_select_ic(timeserie, ic=['aic', 'bic'], trend='nc', max_ar=4, max_ma=4)
        # trend_evaluates.append([i, trend_evaluate.aic_min_order, trend_evaluate.bic_min_order])
        map_pq[key] = trend_evaluate.bic_min_order
map_pq
#%%
trend_model = ARIMA_Model(timeseries.iloc[:, [0]], (map_pq[0][0], 1, map_pq[0][1]))
trend_fit_seq = trend_model.fittedvalues
trend_predict_seq = trend_model.predict(start='2019-12-31', end='2030-12-31', dynamic=True)
print(trend_fit_seq, trend_predict_seq)
#%%
map_trend_fit_seq = {}
map_trend_predict_seq = {}
for key in map_pq:
    print(key)
    indexs=[i for i,x in enumerate(timeseries.iloc[:,key]) if timeseries.iat[i,key]!=0]
    if len(indexs) <= 5:
        continue
    trend = timeseries.iloc[indexs,key]
    trend_model = ARIMA_Model(trend, (map_pq[key][0], 0, map_pq[key][1]))
    trend_fit_seq = trend_model.fittedvalues
    trend_predict_seq = trend_model.predict(start='2019-12-31', end='2030-12-31', dynamic=True)
    map_trend_fit_seq[key] = trend_fit_seq
    map_trend_predict_seq[key] = trend_predict_seq
print(map_trend_fit_seq)
print(map_trend_predict_seq)

#%%
key = 0
# cols=[x for i,x in enumerate(timeseries.columns) if timeseries.iat[0,key]==0]
# timeseries.drop(cols,axis=0)
indexs=[i for i,x in enumerate(timeseries.iloc[:,key]) if timeseries.iat[i,key]!=0]
timeserie = timeseries.iloc[indexs,key]
trend_evaluate = sm.tsa.arma_order_select_ic(timeserie, ic=['aic', 'bic'], trend='nc', max_ar=4, max_ma=4)
trend_evaluate.bic_min_order
#%%
i = 1
trend_model = ARIMA_Model(timeserie, (2, i, 0))
trend_fit_seq = trend_model.fittedvalues
trend_predict_seq = trend_model.predict(start='2019-12-31', end='2030-12-31', dynamic=True)

predict = timeserie.copy()
m = predict.shape
if i == 1:
    base = timeserie.iloc[0]
    for i in range(1, m[0]):
        base = base + trend_fit_seq.iloc[i-1]
        predict.iloc[i] = base
else:
    predict = trend_fit_seq
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
ax1.plot(predict)
ax1.plot(timeserie)
ax2 = fig.add_subplot(212)
ax2.plot(trend_predict_seq)
plt.show()
print(trend_predict_seq)
# %%
# 采用SVR进行预测
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import numpy as np
indexs = [3, 31, 33]
da = pd.DataFrame()
j = {}
for index in indexs:
    clf = SVR(kernel='linear', C=1.25)
    # 三个参数，第一个为上一年历史数据，第二个第三个为组合，对时间参数进行转化，闰年为0，1，常规年份为1,0
    x = [[datas.iat[i-1997-1,31] if i != 0 else 0, \
        0 if (i % 4 == 0 and i % 100 != 0) or (i % 400 == 0) else 1,\
        1 if (i % 4 == 0 and i % 100 != 0) or (i % 400 == 0) else 0] \
        for i in range(1997, 2019)]
    x_tran,x_test,y_train,y_test = train_test_split(x, datas.iloc[:,index].tolist(), test_size=0.1, shuffle= False)
    print(x_tran)
    print(y_train)
    clf.fit(x_tran, y_train)
    y_hat = clf.predict(x_test)


    print("得分:", r2_score(y_test, y_hat))


    r = len(x_test) + 1
    print(y_test)
    plt.plot(np.arange(1,r), y_hat, 'go-', label="predict")
    plt.plot(np.arange(1,r), y_test, 'co-', label="real")
    plt.legend()
    plt.show()

    preictions = []
    test = [[datas.iat[21,index], 1, 0]]
    for i in range(2019, 2031):
        preictions.append(clf.predict(test))
        test = [[preictions[i-2019], \
        0 if (i % 4 == 0 and i % 100 != 0) or (i % 400 == 0) else 1,\
        1 if (i % 4 == 0 and i % 100 != 0) or (i % 400 == 0) else 0]]
    l = []
    for i in preictions:
        l.append(i[0])
    j[index] = l
da = pd.DataFrame(j)
print(da)
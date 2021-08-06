import numpy as np
from scipy.fftpack import fft,ifft
from scipy.signal import stft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
 
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号
 
#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,140)      
 
#设置需要采样的信号，频率分量有200，400和600
y=7*np.sin(2*np.pi*20*x) + 5*np.sin(2*np.pi*40*x)+3*np.sin(2*np.pi*60*x)
 
fft_y=fft(y)                          #快速傅里叶变换
 
N=140
x = np.arange(N)             # 频率个数
half_x = x[range(int(N/2))]  #取一半区间
 
abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
angle_y=np.angle(fft_y)            #取复数的角度
normalization_y=abs_y/N            #归一化处理（双边频谱）                              
normalization_half_y = normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
# cepstrum
ceps = ifft(np.log(np.abs(fft_y))).real
f, t, stft_y = stft(y, 140)
 
plt.subplot(241)
plt.plot(x,y)   
plt.title('原始波形')
 
plt.subplot(242)
plt.plot(x,fft_y,'black')
plt.title('双边振幅谱(未求振幅绝对值)',fontsize=9,color='black') 
 
plt.subplot(243)
plt.plot(x,abs_y,'r')
plt.title('双边振幅谱(未归一化)',fontsize=9,color='red') 
 
plt.subplot(244)
plt.plot(x,angle_y,'violet')
plt.title('双边相位谱(未归一化)',fontsize=9,color='violet')
 
plt.subplot(245)
plt.plot(x,normalization_y,'g')
plt.title('双边振幅谱(归一化)',fontsize=9,color='green')
 
plt.subplot(246)
plt.plot(half_x,normalization_half_y,'blue')
plt.title('单边振幅谱(归一化)',fontsize=9,color='blue')

plt.subplot(247)
plt.plot(x,ceps,'yellow')
plt.title('倒频谱',fontsize=9,color='yellow')

plt.subplot(248)
# plt.plot(t,stft_y,'black')
plt.title('STFT',fontsize=9,color='black')
 # 求幅值
Z = np.abs(stft_y)
# 如下图所示
# plt.pcolormesh(t, f, Z, vmin = 0, vmax = Z.mean()*10)
# plt.plot(half_x,Z,'blue')
plt.show()



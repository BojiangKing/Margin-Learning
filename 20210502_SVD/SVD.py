#%%
import numpy as np


# %%
# test demo1
X = np.random.rand(5, 3)
U, S, V = np.linalg.svd(X, full_matrices=True)
Uhat, Shat, Vhat = np.linalg.svd(X, full_matrices=False)
print(U, Uhat)
print(S, Shat)
print(V, Vhat)
# %%
import matplotlib.pyplot as plt
dirpath = f"C:/Users/FX2/Downloads/1.jpg"
img_plt = plt.imread(dirpath)

# %%
def rebuild_img(u,sigma,v,p):
    m=len(u)
    n=len(v)
    a=np.zeros((m,n))
 
    count=(int)(sum(sigma))
    curSum=0
    k=0
 
    while curSum<=count*p:
        uk=u[:,k].reshape(m,1)
        vk=v[k].reshape(1,n)
        a+=sigma[k]*np.dot(uk,vk)
        curSum+=sigma[k]
        k+=1
 
    a[a<0]=0
    a[a>255]=255
    
    return np.rint(a).astype(int)
# %%
m = len(img_plt)
n = len(img_plt[0])
for i in np.arange(0.1,1,0.1):
    u,sigma,v=np.linalg.svd(img_plt[:,:,0])
    # print(f"U.shape:{U.shape},S.shape:{S.shape},V.shape:{V.shape}")
    R=rebuild_img(u,sigma,v,i)
 
    u,sigma,v=np.linalg.svd(img_plt[:,:,1])
    G=rebuild_img(u,sigma,v,i)
 
    u,sigma,v=np.linalg.svd(img_plt[:,:,2])
    B=rebuild_img(u,sigma,v,i)

    img = [[[R[j,k], G[j,k], B[j,k]] for k in range(n)] for j in range(m)]
 
    # I=np.stack((R,G,B),2)
    # plt.subplot(3,3,0+i*10)
    plt.title(i)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
 

# %%
def svd_truncation(u,sigma,v,r):
    '''truncation value r'''
    m=len(u)
    n=len(v)
    a=np.zeros((m,n))

    if r == -1:
        r = len(sigma)
    
    for k in range(r):
        uk=u[:,k].reshape(m,1)
        vk=v[k].reshape(1,n)
        a+=sigma[k]*np.dot(uk,vk)
 
    a[a<0]=0
    a[a>255]=255
    
    return np.rint(a).astype(int)
# %%
sizes = [5, 20, 100, -1]
l = len(sizes)
for i in range(l):
    if sizes[i] == -1:
        R=img_plt[:,:,0]
        G=img_plt[:,:,1]
        B=img_plt[:,:,2]
    else:
        u,sigma,v=np.linalg.svd(img_plt[:,:,0])
        R=svd_truncation(u,sigma,v,sizes[i])
    
        u,sigma,v=np.linalg.svd(img_plt[:,:,1])
        G=svd_truncation(u,sigma,v,sizes[i])
    
        u,sigma,v=np.linalg.svd(img_plt[:,:,2])
        B=svd_truncation(u,sigma,v,sizes[i])
    
    I=np.stack((R,G,B),2)
    plt.subplot(l,1,i+1)
    # plt.title(i)
    plt.imshow(I)
 
plt.show()
# %%

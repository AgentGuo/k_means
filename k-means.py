#!/usr/bin/env python
# coding: utf-8

# # k-means聚类算法

# In[1]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ## 加载数据

# In[2]:


def load_dataset(filename):
    iris_file = open(filename)
    templist = []
    color = []   # 用于给初始数据画图
    while True:
        line = iris_file.readline()
        linelist = line.strip().replace('.','').split(',')   # 将小数点去掉，等效于将数据整体乘以10
        if len(linelist) != 5:
            break
        if linelist[4] == 'Iris-setosa':     # 给花的种类编号  Iris-setosa：0，Iris-versicolor：1，Iris-virginica：2
            classNum = 0
            color.append([1,0,0])    # 红色
        elif linelist[4] == 'Iris-versicolor':
            classNum = 1
            color.append([0,1,0])   # 绿色
        else:
            classNum = 2
            color.append([0,0,1])   # 蓝色
        templist.append([classNum,int(linelist[0]),int(linelist[1]),int(linelist[2]),int(linelist[3])])    #添加到列表
    return np.array(templist),np.array(color)   # 转化为矩阵


# In[3]:


iris_data, color = load_dataset('iris.data')


# In[4]:


print(iris_data.shape)


# In[5]:


print(iris_data[0])


# ## 查看初始数据

# In[6]:


print('输入前矩阵',iris_data[:,1:].shape)
irisPca = PCA(n_components=2)     # 使用PCA降维便于直观查看初始数据
pcaDate = irisPca.fit_transform(iris_data[:,1:])
print('输入后矩阵',pcaDate.shape)


# In[7]:


X = pcaDate[:,:1].squeeze()
Y = pcaDate[:,1:].squeeze()
print(X.shape, Y.shape, color.shape, X[0], Y[0], color[0])
plt.scatter(X, Y, c=color)
plt.show()


# ## k-means算法

# ### 1.随机选取初始向量

# In[8]:


u1 = iris_data[0][1:]
u2 = iris_data[1][1:]
u3 = iris_data[2][1:]


# In[9]:


print(u1)


# In[10]:


print(iris_data.sum(axis=0)/iris_data.shape[0])


# ### 2.开始循环

# In[11]:


while True:
    C1 = []   # 三个簇集合
    C2 = []
    C3 = []
    for x in iris_data:
        d1 = np.sum(np.square(x[1:] - u1))   # 求样本与每个初始向量的距离的平方（不需要开方）
        d2 = np.sum(np.square(x[1:] - u2))
        d3 = np.sum(np.square(x[1:] - u3))
        min_d = min(d1, d2, d3)
        if d1 == min_d:   # 根据最小距离分配到相应的簇集合
            C1.append(x)
        elif d2 == min_d:
            C2.append(x)
        else:
            C3.append(x)
    C1 = np.array(C1)
    C2 = np.array(C2)
    C3 = np.array(C3)
    #print(C3.shape)
    u1New = np.array([0,0,0,0]) if C1.shape[0] == 0 else C1[:,1:].sum(axis=0)/C1.shape[0]  # 计算新的均值向量
    u2New = np.array([0,0,0,0]) if C2.shape[0] == 0 else C2[:,1:].sum(axis=0)/C2.shape[0]
    u3New = np.array([0,0,0,0]) if C3.shape[0] == 0 else C3[:,1:].sum(axis=0)/C3.shape[0]
    if all(u1New == u1) and all(u2New == u2) and all(u3New == u3):     # 如果均值向量不再更新，则停止循环
        break
    else:
        u1 = u1New
        u2 = u2New
        u3 = u3New
    
    
    # 下面是画图部分
    # 不同的颜色代表不同的簇
    # 三种花分别用'*','x'和'+'表示
    for x in C1:
        if x[0] == 0:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))   # 使用之前训练好的pca降维
            plt.scatter(X, Y, c=[1,0,0], marker = '*')
        elif x[0] == 1:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[1,0,0], marker = 'x')
        else:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[1,0,0], marker = '+')
    for x in C2:
        if x[0] == 0:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[1,0,1], marker = '*')
        elif x[0] == 1:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[1,0,1], marker = 'x')
        else:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[1,0,1], marker = '+')
    for x in C3:
        if x[0] == 0:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[0,0,1], marker = '*')
        elif x[0] == 1:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[0,0,1], marker = 'x')
        else:
            [[X,Y]] = irisPca.transform(x[1:].reshape((1,4)))
            plt.scatter(X, Y, c=[0,0,1], marker = '+')
    # 画出均值向量点
    [[X,Y]] = irisPca.transform(u1.reshape((1,4)))
    plt.scatter(X, Y,s=200 , c=[1,0,0], marker = 'o')
    [[X,Y]] = irisPca.transform(u2.reshape((1,4)))
    plt.scatter(X, Y,s=200 , c=[1,0,1], marker = 'o')
    [[X,Y]] = irisPca.transform(u3.reshape((1,4)))
    plt.scatter(X, Y,s=200 , c=[0,0,1], marker = 'o')
    plt.show()


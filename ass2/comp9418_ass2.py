
# coding: utf-8

# # Assignment 2 (Practical)
# 
# **COMP9418 - Advanced Topics in Statistical Machine Learning**
# 
# **Louis Tiao** (TA), **Edwin V. Bonilla** (Instructor)
# 
# *School of Computer Science and Engineering, UNSW Sydney*
# 
# ---
# 
# In the practical component of this assignment you will build a *class-conditional classifier* using the mixture model described in the theory section of this assignment.
# 
# The basic idea behind a class conditional classifier is to train a separate model for each class $p(\mathbf{x} \mid y)$, and use Bayes' rule to classify a novel data-point $\mathbf{x}^*$ with:
# 
# $$
# p(y^* \mid \mathbf{x}^*) = \frac{p(\mathbf{x}^* \mid y^*) p(y^*)}{\sum_{y'=1}^C p(\mathbf{x}^* \mid y') p(y')}
# $$
# 
# (c.f. Barber textbook BRML, 2012, $\S$23.3.4 or Murphy textbook MLaPP, 2012, $\S$17.5.4).
# 
# In this assignment, you will use the prescribed mixture model for each of the conditional densities $p(\mathbf{x} | y)$ and a Categorical distribution for $p(y)$.
# 
# ### Prerequisites
# 
# You will require the following packages for this assignment:
# 
# - `numpy`
# - `scipy`
# - `scikit-learn`
# - `matplotlib`
# - `observations`
# 
# Most of these may be installed with `pip`:
# 
#     pip install numpy scipy scikit-learn matplotlib observations
# 
# ### Guidelines
# 
# 1. Unless otherwise indicated, you may not use any ML libraries and frameworks such as scikit-learn, TensorFlow to implement any training-related code. Your solution should be implement purely in NumPy/SciPy.
# 2. Do not delete any of the existing code-blocks in this notebook. It will be used to assess the performance of your algorithm.
# 
# ### Assessment
# 
# Your work will be assessed based on:
# - **[50%]** the application of the concepts for doing model selection, which allows you to learn a single model for prediction (Section 1);  
# - **[30%]** the code you write for making predicitions in your model (Section 2); and
# - **[20%]** the predictive performance of your model (Section 3). 

# ## Dataset
# 
# You will be building a class-conditional classifier to classify digits from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), containing grayscale images of clothing items --- coats, shirts, sneakers, dresses and the like.
# 
# This can be obtained with [observations](https://github.com/edwardlib/observations), a convenient tool for loading standard ML datasets.

# In[ ]:

from observations import fashion_mnist
from sklearn.preprocessing import LabelBinarizer
import sklearn.decomposition

# In[ ]:

(x_train, y_train_), _ = fashion_mnist('.')


# There are 60k training examples, each consisting of 784-dimensional feature vectors corresponding to 28 x 28 pixel intensities.

# In[ ]:

# x_train.shape


# The pixel intensities are originally unsigned 8-bit integers (`uint8`) and should be normalized to be floating-point decimals within range $[0,1]$.

# In[ ]:

x_train = x_train / 255.
x_train=x_train[:2000]
reducer=sklearn.decomposition.PCA(n_components=100)
reducer.fit(x_train)
x_train=reducer.transform(x_train)
print (x_train[0],x_train[1])

# The targets contain the class label corresponding to each example. For this assignment, you should represent this using the "one-hot" encoding. 

# In[ ]:

y_train = LabelBinarizer().fit_transform(y_train_)
# y_train.shape


# Note that you are only to use the training data contained in `x_train`, `y_train` as we have define it. In order to learn and test you model, you may consider splitting these data into training, validation and testing. You may not use any other data to for training.
# 
# In particular, if you want to assess the performance of your model in section 2, you must create a test set `test.npz`. You are not required to submit this test file as we will evaluate the performance of your model using our own test data.

# ## Preamble 

# In[ ]:




# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import random

# #### Constants

# You can use the function below to plot a digits in the dataset.

# In[ ]:

def plot_image_grid(ax, images, n=20, m=None, img_rows=28, img_cols=28):
    """
    Plot the first `n * m` vectors in the array as 
    a `n`-by-`m` grid of `img_rows`-by-`img_cols` images.
    """
    if m is None:
        m = n
 
    grid = images[:n*m].reshape(n, m, img_rows, img_cols)

    return ax.imshow(np.vstack(np.dstack(grid)), cmap='gray')


# Here we have the first 400 images in the training set.

# In[ ]:

# fig, ax = plt.subplots(figsize=(8, 8))
#
# plot_image_grid(ax, x_train, n=20)
#
# plt.show()
#
#
# # Here we have the first 400 images labeled "t-shirts" in the training set.
#
# # In[ ]:
#
# fig, ax = plt.subplots(figsize=(8, 8))
#
# plot_image_grid(ax, x_train[y_train_ == 0])
#
# plt.show()


# ## Section 1 `[50%]`: Model Training
# 
# Place all the code for training your model using the function `model_train` below. 
# 
# - We should be able to run your notebook (by clicking 'Cell->Run All') without errors. However, you must save the trained model in the file `model.npz`. This file will be loaded to make predictions in section 2 and assess the performance of your model in section 3. Note that, in addition to this notebook file, <span style="color:red"> ** you must provide the file `model.npz` **</span>.
# 
# - You should comment your code as much as possible so we understand your reasoning about training, model selection and avoiding overfitting. 
# 
# - You can process the data as you wish, e.g. by applying some additional transformations, reducing dimensionality, etc. However, all these should be here too. 
# 
# - Wrap all your training using the function `model_train` below. You can call all other custom functions within it.

# In[ ]:
def Gaussian(data,mean,cov):
    dim = np.shape(data)[0]
    # print('weidu',dim)# 计算维度!!!!!!!!!!!!!!!!!
    covdet = np.linalg.det(cov) # 计算|cov|
    # print(covdet)
    covinv = np.linalg.inv(cov) # 计算cov的逆
    # print(covinv.shape)
    if covdet==0.0 and covdet==-0.0:              # 以防行列式为0
        covdet = np.linalg.det(cov+np.eye(dim)*0.01)
        covdet = 1
        covinv = np.linalg.inv(cov+np.eye(dim)*0.01)
        ##矩阵求逆
    diff = data - mean
    z = -0.5 * np.multiply(np.multiply(diff, covinv),diff.T)
    # print('z',np.power(2*np.pi,784))# 计算exp()里的值
    return 1.0 / (np.power(np.power(2 * np.pi, dim) * abs(covdet), 0.5)) * np.exp(z)


def GetInitialMeans(data,K):
    dim = data.shape[1]  # 数据的维度
    means = [[] for k in range(K)]
    # 存储均值
    minmax=[]
    for i in range(dim):
        minmax.append(np.array([min(data[:,i]),max(data[:,i])]))  # 存储每一维的最大最小值
        # print('每一维的最大最小值', minmax)
    minmax=np.array(minmax)
    for i in range(K):
        means[i]=[]
        for j in range(dim):
            means[i].append(np.random.random()*(minmax[i][1]-minmax[i][0])+minmax[i][0] ) #随机产生means,每一维最大减最小乘个0.几的随机数，再加上最小
        means[i]=np.array(means[i])
            # print(means[i])
    return means
# pis=np.array([0.1]*784).reshape(1,-1)
# X=np.array(x_train)
# mu=np.array(np.random.normal(loc=0, scale=0.1, size=10*784)).reshape([10,784,1])
# cov=np.diag(pis[0])
# def calGassuianProb(X,mu,cov):
#     mu = np.array(mu).reshape((-1,1))  # Dx1
#     D = len(mu)
#     cov = np.array(cov).reshape(D,D)
#     X_mu = X - mu  # DxN
#     return np.exp ( - (np.log(np.linalg.det(cov)) + np.diag(np.matmul(np.matmul(X_mu.T,inv(cov)),X_mu)) + D*np.log(2*np.pi)) / 2 )
# a=calGassuianProb(X,mu,cov)
# print(a)

def model_train(x_train, y_train):
    """
    Write your code here.
    """
    data=x_train
    D,N=data.shape##60000,784
    # print(D,N)
    # print(data.shape)
    # print(y_train.shape)
    K=y_train.shape[1]###10类
    pai=np.array([1]*K)/K
    Q=30
    bias=0.0000000000001


    Z=np.array(np.random.normal(loc=0,scale=0.1,size=Q).reshape([Q,1]))##对于隐变量
    miu=np.array(np.random.normal(loc=0, scale=0.1, size=K*N)).reshape([K,N,])
    W = np.array(np.random.normal(loc=0, scale=0.1, size=K*N*Q)).reshape([K,N,Q])
    psi=np.eye(N)
    # print ('QWEQWEQWE',psi.shape)
    # print('fangcha',psi.shape)
    # convs=np.linalg.inv(pis)
    # print(convs)
    # for i in range(K):
    #     convs[i] = np.cov(data.T)
    # # convs[np.eye(5)] for k in range(K)
    # print('方差的维度',convs.shape())
    # #E
    # rnk = [np.zeros(K) for i in range(N)]
    # print(gammas[0][0],gammas[N-1][K-1])
    r=[]
    R=[]
    newloglikelyhood=0
    oldloglikelyhood=1
    rnk = np.array([np.zeros(K) for i in range(N)])###初始rnk表
    print (rnk.shape)
    # while np.abs(oldloglikelyhood-newloglikelyhood)>0.0001:  ###10类
    while True:
        print ('****')
        sum=0
        ##-----------EEEE-step----------------##
        for i in range(N):
            rnk_list=[]
            for k in range(K):
                # print ('Wk',W[k].shape)
                tem=psi + np.matmul(W[k] , W[k].T)
                # print (np.count_nonzero(tem))
                if np.linalg.det(tem)==0:
                    tem = np.where(tem == 0, bias, tem)
                    # tem[0][0]=tem[0][0]+bias*0.01
                    # print (tem[0][0])
                    # print (np.count_nonzero(tem))

                else:
                    tem=tem

                beta_k = np.matmul(W[k].T , np.linalg.inv(tem))
                #print ('beta', beta_k)
                E = np.sum( pai[k]* Gaussian(data[i],data[i]-miu[k], (np.matmul(W[k],W[k].T)+psi)))
                # print ('EEEEEEE',E)
                # np.random.rand(1)
                rnk_list.append(E)
                # print(miu[k].shape)
                Ez_w__x=np.matmul(beta_k,(data[i]-miu[k]))
                diff=data[i]-miu[k]
                Ez_w__x=Ez_w__x.reshape(Ez_w__x.shape[0],1)
                diff=diff.reshape(data[i].shape[0],1)

                # print ('E1',Ez_w__x)
                line_one = np.ones(shape=(1,1))
                Ez_w__x_2=np.vstack((Ez_w__x,line_one))
                # print ('AD1',Ez_w__x_2.shape)
                #print (np.matmul(beta_k,(data[i]-miu[k])))
                Ewzzt_x=(np.identity(Q)-np.matmul(beta_k,W[k])+np.matmul(np.matmul(np.matmul(beta_k,diff),diff.T),beta_k.T))
                # print ('E2',Ewzzt_x.shape)
                Ewzzt_x2=np.column_stack((np.row_stack((Ewzzt_x,Ez_w__x.T)),Ez_w__x_2))
                # print ('AD2',Ewzzt_x2.shape)
                # Q=0-p1-p2
                # print ('sadsadsa',np.linalg.det(psi))

                W_k=np.column_stack((W[k],np.ones(shape=[diff.shape[0],line_one.shape[1]])))

                # print ('WKKK',W_k.shape)
                # print ('psi',psi.shape)
                if np.linalg.det(psi)==0:
                    psi1=np.where(psi==0,bias,psi)
                    # psi[0][0]=psi[0][0]+bias
                else:
                    psi1=psi
                xx=np.matmul(np.matmul(np.matmul(W_k.T,np.linalg.inv(psi1)),W_k),Ewzzt_x2)
                p4=0.5*rnk[i][k]*np.trace(xx)
                # print ('PPPPP4',p4)
                # print (data[0])
                # print (np.matmul(data[i].T,psi))
                p2=-0.5*rnk[i][k]*np.matmul(np.matmul(data[i].T,np.linalg.inv(psi)),data[i])
                # print ('PPPP2',p2)
                p3=-rnk[i][k]*np.matmul(np.matmul(np.matmul(data[i].T,np.linalg.inv(psi)),W_k),Ez_w__x_2)
                # print ('PPPPP3',p3)
                #jia 1
                sum=p2+p3+p4+sum
            # sumres=np.sum(rnk_list)  ##求rnk的概率和
            # for k in range(K):###归一，做N个样本属于K个类的概率
                rnk[i][k]=rnk_list[k]
        p1 = -N / 2 * np.log(abs(np.linalg.det(psi)))
        # print ('PPPPP1', -p1)
        # newloglikelyhood=0.1-p1+sum
        # print ('SUM',Q)
        ##--------M-step----------------########
        W_k_p1_sum=0
        Mu_k_p1_sum=0

        pai_new_list = []
        for k in range(K):
        ##更新【W，均值】
            ##跟新pai 对i求和
            pai_new_sum=0
            W_k_news=[]
            for i in range(N):

                tem = psi + np.matmul(W[k], W[k].T)
                if np.linalg.det(tem) == 0:
                    tem = np.where(tem == 0, bias, tem)
                    # tem[0][0] = tem[0][0] + bias * 0.01
                else:
                    tem = tem

                beta_k = np.matmul(W[k].T, np.linalg.inv(tem))
                # print ('beta', beta_k)
                # rnk = np.sum( pai[k]* Gaussian(data[i],data[i]-miu[k], (np.matmul(W[k],W[k].T)+psi)))

                # print(miu[k].shape)
                Ez_w__x = np.matmul(beta_k, (data[i] - miu[k]))
                diff = data[i] - miu[k]
                # print ('DATA',data[i].shape)
                data_i=data[i]
                data_i = data_i.reshape(data_i.shape[0], 1)
                # print ('DATA',data_i.shape)
                Ez_w__x = Ez_w__x.reshape(Ez_w__x.shape[0], 1)
                diff = diff.reshape(diff.shape[0], 1)
                # print ('E1', Ez_w__x.shape)
                line_one = np.ones(shape=(1, 1))
                Ez_w__x_2 = np.vstack((Ez_w__x, line_one))
                # print ('AD1',Ez_w__x_2.shape)
                Ewzzt_x = (np.identity(Q) - np.matmul(beta_k, W[k]) + np.matmul(np.matmul(np.matmul(beta_k, diff), diff.T), beta_k.T))
                # print ('E2', Ewzzt_x.shape)
                Ewzzt_x2 = np.column_stack((np.row_stack((Ewzzt_x, Ez_w__x.T)), Ez_w__x_2))
                # print ('AD',Ewzzt_x2.shape)
                W_k_p1_sum=rnk[i][k]*np.matmul(data_i,Ez_w__x_2.T)+W_k_p1_sum
                Mu_k_p1_sum=rnk[i][k]*Ewzzt_x2+Mu_k_p1_sum
                ###pai的加和
                # print ('RNK',rnk[i][k])
                pai_new_sum=rnk[i][k]+pai_new_sum
            pai_ave=pai_new_sum/N   #####更新PAI
            pai_new_list.append(pai_ave)
            pai=np.array(pai_new_list)
            # print ('PPPAAAAAIII',pai)
            W_k_new=np.matmul(W_k_p1_sum,np.linalg.inv(Mu_k_p1_sum))
            # print ('一个NEW',W_k_new.shape)
            W_k_news.append(W_k_new)
            W[k,:,:]=W_k_new[:,:W_k_new.shape[1]-1]
            # print ('XIN WWW',W.shape)####更新WWWWW
            miu[k,:]=W_k_new[:,-1].T  ####更新MIU!!
            # print ('MIU',miu.shape)
            # print ("KKKKK新",W_k_new.shape)
            # print ('MUMUMU新',miu.shape)
            # print ('MIU的维度',np.linalg.inv(Mu_k_p1_sum).shape)

            # print('WEIDU',W_k_new.shape)

        # print ('HEHE', np.sum(pai_new_list))
        ##更新协方差矩阵
        psi_new_p0=0
        ##对i求和
        for i in range(N):
            ##对 k求和，
            psi_new_p1=0
            for k in range(K):


                data_i = data[i]
                data_i = data_i.reshape(data_i.shape[0], 1)
                tem = psi + np.matmul(W[k], W[k].T)
                if np.linalg.det(tem) == 0:
                    tem = np.where(tem == 0, bias, tem)
                    tem[0][0] = tem[0][0] + bias * 0.01
                else:
                    tem = tem

                beta_k = np.matmul(W[k].T, np.linalg.inv(tem))
                # print ('beta', beta_k)
                # rnk = np.sum( pai[k]* Gaussian(data[i],data[i]-miu[k], (np.matmul(W[k],W[k].T)+psi)))

                Ez_w__x = np.matmul(beta_k, (data[i] - miu[k]))
                diff = data[i] - miu[k]
                # print ('DATA', data[i].shape)
                data_i = data[i]
                data_i = data_i.reshape(data_i.shape[0],1)
                # print ('DATA', data_i.shape)
                Ez_w__x = Ez_w__x.reshape(Ez_w__x.shape[0], 1)
                diff = diff.reshape(diff.shape[0], 1)
                # print ('E1', Ez_w__x.shape)
                line_one = np.ones(shape=(1, 1))
                Ez_w__x_2 = np.vstack((Ez_w__x, line_one))
                # print ('AD1', Ez_w__x_2.shape)
                # print (np.matmul(np.column_stack((W[k],miu[k])),Ez_w__x_2).shape)
                p1=(np.matmul(np.column_stack((W[k],miu[k])),Ez_w__x_2))
                # print ('P1',p1.shape,rnk[i][k])
                psi_new_p1=rnk[i][k]*np.matmul((data_i-p1),data_i.T)+psi_new_p1
            psi_new_p0=psi_new_p1+psi_new_p0
        ##最后的取对角线得新的协方差矩阵
        # print ('%%%%%%%',psi_new_p0.shape)
        #####见论文
        psi=np.diag(np.diag(psi_new_p0)) # 更新方差
        print (psi[0][0])
        # print ('PPPSSSII',Psi_New,np.trace(psi_new_p0))
        # rnk_=rnk/sumres
        #     r.append(np.sum(rnk))##????????????
        # print('每一行数据的和', r)
        # # print('dasdas',len(r))
        # R.append(r)
    # print(np.array(R)[49])
    ##计算Q（log_likelihood）

        # print ('NEWLOG', newloglikelyhood)

    # const=-N/2*log(np.linalg.det(psi))
    # part2=0
    # # part3=
    # for i in range(N):
    #     for j in range(K):
    #         part2=0.5*rnk*data[i].T*np.linalg.inv(psi)*data[i]+part2

    model = None

    # You can modify this to save other variables, etc 
    # but make sure the name of the file is 'model.npz.
    np.savez_compressed('model.npz', model=model)


# ## Section 2 `[30%]`: Predictions
# 
# Here we will assume that there is a file `test.npz` from which we will load the test data.  As this file is not given to you, you will need to create one yourself (but not to submit it) to test your code. <span style="color:red">Note that if you do not create this file the cell below will not run</span>. 
# 
# Your task is to fill in the `model_predict` function below. Note that this function should load your `model.npz` file, which must contain all the data structures necessary for making predictions.

# In[ ]:

# create these yourself for your own testing but need to delete before submisson
# x_test = np.random.randn(10000, 784)
# y_test = np.random.randint(low=0, high=9, size=(10000,1))
# # y_test.shape
# np.savez('test.npz', x_test=x_test, y_test=y_test)
#
#
# # In[ ]:
#
# test = np.load('test.npz')
# x_test = test.get('x_test')
# y_test = test.get('y_test')


# In[ ]:

# x_test.shape


# In[ ]:

# y_test_ = LabelBinarizer().fit_transform(y_test)
# y_test_.shape


# In[ ]:

# fig, ax = plt.subplots()
#
# plot_image_grid(ax, x_test, n=8, m=3)
#
# plt.show()


# In[ ]:

# def model_predict(x_test):
#     """
#     @param x_test: (N_test,D)-array with test data
#     @return y_pred: (N,C)-array with predicted classes using one-hot-encoding
#     @return y_log_prob: (N,C)-array with  predicted log probability of the classes
#     """
#
#     # Add your code here: You should load your trained model here
#     # and write to the corresponding code for making predictions
#     model = np.load('model.npz')
#
#     return y_pred, y_log_prob


# ## Section 3 `[20%]`: Performance 
# 
# You do not need to do anything in this section but you can use it to test the generalisation performance of your code. We will use it the evaluate the performance of your algorithm on a new test. 

# In[ ]:

# def model_performance(x_test, y_test, y_pred, y_log_prob):
#     """
#     @param x_test: (N,D)-array of features
#     @param y_test: (N,C)-array of one-hot-encoded true classes
#     @param y_pred: (N,C)-array of one-hot-encoded predicted classes
#     @param y_log_prob: (N,C)-array of predicted class log probabilities
#     """
#
#     acc = np.all(y_test == y_pred, axis=1).mean()
#     llh = y_log_prob[y_test == 1].mean()
#
#     return acc, llh


# In[ ]:

# y_pred, y_log_prob = model_predict(x_test)
# acc, llh = model_performance(x_test, y_test, y_pred, y_log_prob)


# In[ ]:

# 'Average test accuracy=' + str(acc)
#
#
# # In[ ]:
#
# 'Average test likelihood=' + str(llh)




if __name__ == "__main__":
    print ('xx')
    x=model_train(x_train, y_train)
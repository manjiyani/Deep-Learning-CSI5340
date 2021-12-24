#results stored in the non-regularized folder
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def getData(sd, N):
    X = np.random.uniform(0,1,N)
    Z = np.random.normal(0,sd,N)
    Y = np.cos(2*math.pi*X) + Z
    return np.vstack((X,Y)).T

def getMSE(y_pred,y):
    mse = np.mean((y-y_pred)**2)
    return mse

def GD(X,y,n,weights):
    itr = 2000
    lrate = 0.001
    Ein_list = np.zeros(itr)
    Ein=0
    for i in range(itr):
        weights -= lrate * (2/n) * (((X.dot(weights) - y).T).dot(X).T)
        Ein_list[i] = getMSE(X.dot(weights), y)
    Ein = Ein_list[itr-1]
    return weights,Ein

def SGD(X,Y,n,weights):
    itr = 2000
    lrate = 0.001
    Ein_list = np.zeros(itr)
    Ein=0
    for i in range(itr):
        random_index = random.randint(0,n-1)
        x=X[random_index].reshape(-1,1)
        y=Y[random_index].reshape(-1,1)
        weights -= lrate * (2/n) * (((x*weights - y))*x)
        Ein_list[i] = getMSE(X.dot(weights), Y)
    Ein = Ein_list[itr-1]
    return weights,Ein

def Mini_batch(X,Y,n,weights):
    itr = 2000
    lrate = 0.001
    batch_size=50
    Ein_list = np.zeros(itr)
    Ein=0
    for i in range(itr):
        if(batch_size>=n):
            x=X
            y=Y
            batch_size=n
        else:
            random_index = random.randint(0,n-batch_size)
            x=X[random_index:random_index+batch_size]
            y=Y[random_index:random_index+batch_size]
        weights -= lrate * (2/batch_size) * ((x.dot(weights) - y).T.dot(x).T)
        Ein_list[i] = getMSE(X.dot(weights), Y)
    Ein = Ein_list[itr-1]
    return weights,Ein

def fitData(data,d,setter):
    y = data[:,1].reshape((len(data),1))
    x = data[:,0]
    X = np.ones((1,len(data)))
    for i in range(d):
        X = np.vstack((X, x**(i+1)))
    X= X.T
    #print(X)
    
    #initialize weights
    weights = np.random.random([d+1,1])
    #print(weights)
    #print(((X.dot(weights) - y).T).dot(X).T)
    n=len(data)

    #function setup
    if(setter==1):
        weights,Ein = GD(X,y,n,weights)
    elif(setter ==2):    
        weights,Ein = SGD(X,y,n,weights)
    else:
        weights,Ein = Mini_batch(X,y,n,weights)
    
    
    return weights,Ein

def testing(tdata,weights,d):
    n=len(tdata)
    y=tdata[:,1].reshape((n,1))
    x = tdata[:,0]
    X = np.ones((1,n))
    for i in range(d):
        X = np.vstack((X, x**(i+1)))
    X=X.T
    Eout = getMSE(X.dot(weights),y)
    return Eout


def experiment(N,sd,d,setter):
    m=50
    Ein_list=[]
    Eout_list=[]
    weight_list=[]
    for i in range(m):
        tempdata = getData(sd,N)
        temp_weight,temp_Ein = fitData(tempdata,d,setter)
        temp_test_data = getData(sd,1000)
        temp_Eout = testing(temp_test_data,temp_weight,d)
        Ein_list.append(temp_Ein)
        Eout_list.append(temp_Eout)
        weight_list.append(temp_weight)

    Ein_avg = np.mean(Ein_list)
    Eout_avg = np.mean(Eout_list)
    weights_avg = np.mean(weight_list,axis=0)

    #Ebias calc starts here
    tempdata=getData(sd,2000)
    Ebias = testing(tempdata,weights_avg,d)
    Egen = abs(Ein_avg-Eout_avg)

    return Ein_avg,Eout_avg,Ebias



"""
#Can use this section to get Ein, Eout, Ebias for all possible combinations of N, d and sd.
#Result is stored in a 3d list and can be accessed as resulst[n][d][sd]

N=[2,5,10,20,50,100,200]
D= [_ for _ in range(21)]
SD = [0.01,0.1,1]

results=[[[ [] for x in range(len(N))] for x in range(len(D))]for x in range(len(SD))]

for i in  range(len(N)):
    for j in range(len(D)):
        for k in range(len(SD)):
            Einav,Eoutav,Ebias = experiment(N[i],SD[k],D[j])
            results[i][j][k].append(Einav)
            results[i][j][k].append(Eoutav)
            results[i][j][k].append(Ebias)

print(results)
"""

GD_list=[[],[],[]]
SGD_list=[[],[],[]]
Mini_list=[[],[],[]]


#variable complexity
for d in range(1,21):
    #t1,t2,t3 = experiment(200,0.1,d,1)
    #GD_list[0].append(t1)
    #GD_list[1].append(t2)
    #GD_list[2].append(t3)

    #t1,t2,t3 = experiment(200,0.1,d,2)
    #SGD_list[0].append(t1)
    #SGD_list[1].append(t2)
    #SGD_list[2].append(t3)

    t1,t2,t3 = experiment(100,0.5,d,3)
    Mini_list[0].append(t1)
    Mini_list[1].append(t2)
    Mini_list[2].append(t3)

d=[x for x in range(1,21)]
#plt.plot(d,GD_list[0],color='r',linestyle='solid',label='GD-Ein')
#plt.plot(d,GD_list[1],color='r',linestyle='dotted',label='GD-Eout')
#plt.plot(d,GD_list[2],color='r',linestyle='dashdot',label='GD-Ebias')

#plt.plot(d,SGD_list[0],color='y',linestyle='solid',label='SGD-Egen')
#plt.plot(d,SGD_list[1],color='y',linestyle='dotted',label='SGD-Ebias')
#plt.plot(d,SGD_list[2],color='y',linestyle='dashdot',label='SGD-Ebias')

plt.plot(d,Mini_list[0],color='r',linestyle='solid',label='Mini-Ein')
plt.plot(d,Mini_list[1],color='g',linestyle='dotted',label='Mini-Eout')
plt.plot(d,Mini_list[2],color='b',linestyle='dashdot',label='Mini-Ebias')

plt.plot(d[Mini_list[0].index(min(Mini_list[0]))],min(Mini_list[0]),'*')
plt.plot(d[Mini_list[1].index(min(Mini_list[1]))],min(Mini_list[1]),'*')
plt.plot(d[Mini_list[2].index(min(Mini_list[2]))],min(Mini_list[2]),'*')

plt.title('Ein, Eout, Ebias of Mini-batch with variable complexity no weight decay')
plt.xlabel('d (Complexity)')
plt.ylabel('mse')
plt.legend()
plt.show()

"""
#Variable N
N=[2,5,10,20,50,100,200]

for n in N:
    t1,t2,t3 = experiment(n,0.1,50,3)
    Mini_list[0].append(t1)
    Mini_list[1].append(t2)
    Mini_list[2].append(t3)

plt.plot(N,Mini_list[0],color='r',linestyle='solid',label='Mini-Ein')
plt.plot(N,Mini_list[1],color='g',linestyle='dotted',label='Mini-Eout')
plt.plot(N,Mini_list[2],color='b',linestyle='dashdot',label='Mini-Ebias')

plt.title('Ein, Eout, Ebias of Mini-batch with variable dataset size no weight-decay')
plt.xlabel('N (dataset size)')
plt.ylabel('mse')
plt.legend()
plt.show()
"""

"""
#Variable Noise level
SD = [0.01,0.1,0.5,1]

for sd in SD:
    t1,t2,t3 = experiment(100,sd,3,3)
    Mini_list[0].append(t1)
    Mini_list[1].append(t2)
    Mini_list[2].append(t3)
    
plt.plot(SD,Mini_list[0],color='r',linestyle='solid',label='Mini-Ein')
plt.plot(SD,Mini_list[1],color='g',linestyle='dotted',label='Mini-Eout')
plt.plot(SD,Mini_list[2],color='b',linestyle='dashdot',label='Mini-Ebias')

plt.title('Ein, Eout, Ebias of Mini-batch with variable noise')
plt.xlabel('sd (Standard deviation ~ noise level)')
plt.ylabel('mse')
plt.legend()
plt.show()
""" 

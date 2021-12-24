import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    return (1.0/(1.0 + np.exp(x)))

def d_sigmoid(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def L(x,A,B):
    y = A*x
    u = sigmoid(y)
    v = B*x
    z = A*np.multiply(u,v)
    w = A*z
    
    norm = []
    for i in w:
        u = np.linalg.norm(i)
        norm.append(u)
    l = np.power(norm,2)
    loss = sum(l)
    return (loss)

def gradient_cal(x,A,B):
    y = A*x
    u = sigmoid(y)
    v = B*x
    z = A*np.multiply(u,v)
    w = A*z
    
    n= tf.transpose(x,perm=[0,2,1])
    sess = tf.Session()
    x_trans = sess.run(n)
    gra_a = x_trans
    gra_b = x_trans
    gra_y = d_sigmoid(y)
    gra_u = np.multiply(A,v)
    gra_v = np.multiply(A,u)
    gra_z = Trans(A)
    gra_w = 2*w
    gradient_A = gra_w * np.multiply(gra_z,gra_u * np.multiply(gra_a,gra_y))
    gradient_B = gra_w * np.multiply(gra_z,np.multiply(gra_v,gra_b))
    return gradient_A, gradient_B

def Trans(A):
    e=[]
    a = np.array(A)
    for i in a:
        e.append(i.T)
    return e

def backprop(gra_a,gra_b,A,B,x):
    alpha=0.5
    new_a = A + alpha*gra_a
    new_b = B + alpha*gra_b
    new_loss = L(x,new_a,new_b)
    return new_loss, new_a, new_b

def gen_x(N, dim):
    np.random.seed(0)
    x = np.random.randint(10, size=(N,dim,1))
    x= np.array(x)
    return x

def ans_func(N,dim):
    losses=[]
    itr_list=[]
    x = gen_x(N,dim)
    np.random.seed(1)
    A = np.random.randint(0.5,10,size=[dim,dim])
    B = np.random.randint(0.5,10,size=[dim,dim])
    A_sup=[]
    B_sup=[]
    for i in range(N):
        A_sup.append(A)
        B_sup.append(B)
    
    
    for itr in range(1000):
        loss = L(x,A,B)
        gra_A, gra_B = gradient_cal(x,A,B)
        new_loss, new_a, new_b = backprop(gra_A, gra_B, A, B, x)
        A = new_a
        B = new_b
        loss = new_loss
        losses.append(loss)
        itr_list.append(itr)
    
    return loss,losses,itr_list

#Main
loss,losses, itr_list = ans_func(10,5)
plt.figure(figsize=(12,10))
plt.plot(itr_list,losses,label='Loss')
plt.title("Loss value over 1000 iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss Value")
plt.legend()
plt.show()

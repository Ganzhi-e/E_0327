#开发时间：2022/4/11 15:23
import torch
from torch import nn
from d2l import torch as d2l

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size);

num_inputs=784
num_outputs=10
num_hiddens=256

#参数初始化
W1=torch.randn(num_inputs,num_hiddens,requires_grad=True)
b1=torch.zeros(num_hiddens,requires_grad=True)
W2=torch.randn(num_hiddens,num_outputs,requires_grad=True)
b2=torch.zeros(num_outputs,requires_grad=True)
params=[W1,b1,W2,b2]

#激活函数
def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X,a)

#Model
def net(X):
    X=X.reshape((-1,num_inputs))
    H=relu(X@W1+b1)
    return H@W2+b2

#loss
loss=nn.CrossEntropyLoss()

#train
num_epochs=10
lr=0.1
updater=torch.optim.SGD(params,lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)





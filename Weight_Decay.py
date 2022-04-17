#开发时间：2022/4/11 19:54
"""
第一行是lambda声明，x,y相当于传入的参数，整个函数会返回x+y的值。lambda作为一个表达式，定义了一个匿名函数，上例的代码x，y为入口参数，x+y为函数体。在这里lambda简化了函数定义的书写形式。
python允许用lambda关键字创造匿名函数。匿名是不需要以标准的方式来声明，比如说使用 def 语句。（除非赋值给一个局部变量，这样的对象也不会在任何的名字空间内创建名字，上面的例子中会创建名字。)
作为函数，它们也能有参数。一个完整的 lambda"语句"代表了一个表达式，这个表达式的定义体必须和声明放在同一行。语法如下：
lambda [arg1[, arg2, … argN]]: expression
参数是可选的，如果使用的参数话，参数通常也是表达式的一部分
"""
import torch
from torch import nn
from d2l import torch as d2l

#生成数据
n_train,n_test,num_inputs,batch_size=20,100,200,5
true_w,true_b=torch.ones((num_inputs,1))*0.01,0.05
train_data=d2l.synthetic_data(true_w,true_b,n_train)
train_iter=d2l.load_array(train_data,batch_size)
test_data=d2l.synthetic_data(true_w,true_b,n_test)
test_iter=d2l.load_array(test_data,batch_size,is_train=False)


#初始化参数
def init_params():
    w=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return [w,b]

#L2惩罚
def l2_penalty(w):
    return torch.sum(w**2)/2


#训练函数
# def train(lambd):
#     w,b=init_params()
#     net,loss=lambda X: d2l.linreg(X,w,b),d2l.squared_loss
#     num_epochs=100
#     lr=0.003
#     animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
#                             xlim=[5, num_epochs], legend=['train', 'test'])
#     for epoch in range(num_epochs):
#         for X,y in train_iter:
#             l=loss(net(X),y)+lambd*l2_penalty(w)
#             l.sum().backward()
#             d2l.sgd([w,b],lr,batch_size)
#             if (epoch + 1) % 5 == 0:
#                 animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
#                                          d2l.evaluate_loss(net, test_iter, loss)))
#     print('w的L2范数是：', torch.norm(w).item())
# train(0)

def train_concise(wd):
    net=nn.Sequential(nn.Linear(num_inputs,1))
    #参数初始化
    for param in net.parameters():
        param.data.normal_()
    num_epochs=100
    lr=0.003
    loss=nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=wd)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            trainer.zero_grad()#使用框架内optimizer 梯度清零
            l=loss(net(X),y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(net, train_iter, loss),d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
train_concise(10)
d2l.plt.show()
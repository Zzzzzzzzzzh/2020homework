import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## 构造列标签名字,一共11个
#column = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

# 读取数据
data = pd.read_csv(r"G:\python_work\machine learning\classfication\data.csv")

#良性为0，恶性为1
data.diagnosis = data.diagnosis.map({'M':1, 'B':0})
print(data)

# 缺失值进行处理
#data = data.replace(to_replace='?', value=np.nan)
#data = data.dropna()
data.drop(['id'], axis = 1, inplace = True)
print(data)

# 进行数据的分割，2到32列为特征值，1列为目标值
x_train, x_test, y_train, y_test = train_test_split(np.array(data.iloc[:,1:]), np.array(data.iloc[:,0]), test_size=0.3)

# 标准化处理
std = StandardScaler()
x_train = std.fit_transform(x_train) #找到数据的均值方差并标准化
x_test = std.transform(x_test)  #用上一步得到的参数标准化

# 转化为variable类型
x_data = Variable(torch.Tensor(x_train))
y_data = Variable(torch.Tensor(y_train))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(30, 1) # 30 in and 1 out
        
    def forward(self, x):
        #y_pred = torch.sigmoid(self.linear(x))
        y_pred = self.linear(x).sigmoid()
        return y_pred
    
# Our model    
model = Model()

criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)
    
    # Compute and print loss
    # predict = out.ge(0.5).float()   #以0.5为阈值进行分类
    accuracy = 0
    pred=y_pred.detach().numpy()
    if epoch==999:
        accracy1=0
    for i in range(1,y_train.shape[0]):
        if (pred[i]-0.5)*(y_train[i]-0.5)>=0:
            accuracy += (1 / y_train.shape[0])  
    loss = criterion(y_pred, y_data)
    print('epoch:',epoch, 'loss:',loss.data.item(),'accuracy:',accuracy)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for f in model.parameters():
    print('data is')
    print(f.data)
    print(f.grad)

w = list(model.parameters())
w0 = w[0].data.numpy()
w1 = w[1].data.numpy()

#import matplotlib.pyplot as plt

#print("Final gradient descend:", w)
## plot the data and separating line
#plt.scatter(X[:,0], X[:,1], c=T.reshape(N), s=100, alpha=0.5)
#x_axis = np.linspace(-6, 6, 100)
#y_axis = -(w1[0] + x_axis*w0[0][0]) / w0[0][1]
#line_up, = plt.plot(x_axis, y_axis,'r--', label='gradient descent')
#plt.legend(handles=[line_up])
#plt.xlabel('X(1)')
#plt.ylabel('X(2)')
#plt.show()

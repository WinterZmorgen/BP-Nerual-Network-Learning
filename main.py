import xlrd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras

# 读取excel数据
data = xlrd.open_workbook('Data.xls')
sheet = data.sheets()[0]

nrows = sheet.nrows
ncols = sheet.ncols
# 训练集长度
trainset_len = 44

X = []
for i in range(1, nrows):
    X.append(sheet.row_values(i))

# 打乱数据
X = np.array(X)
np.random.seed(10)
np.random.shuffle(X)

# 归一化
for i in range(1, 4):
    max_X = max(X[:, i])
    min_X = min(X[:, i])

    X[:, i] = 0.1 + 0.8 * (X[:, i] - min_X) / (max_X - min_X)

#数据分训练集和测试集
train_X=X[0:trainset_len,0:4]
train_Y=X[0:trainset_len,4:]

print(train_X[0,:])

test_X=X[trainset_len:nrows-1,0:4]
test_Y=X[trainset_len:nrows-1,4:]

#神经网络模型建立
model = keras.Sequential()
model.add(keras.layers.Dense(9, activation='relu', input_dim=4))
model.add(keras.layers.Dense(9, activation='relu'))
model.add(keras.layers.Dense(2))

# #动量梯度下降
# optimizer = keras.optimizers.SGD(learning_rate=0.01,momentum=0.9)
# #RMSprop
# optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
#Adam
optimizer = keras.optimizers.Adam(learning_rate=0.01)

#优化器为SGD   mse为均方误差
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mse'])

#显示网络结构
model.summary()
#进行训练
history = model.fit(train_X, train_Y,epochs=200,verbose=1)

#计算误差
test_predictions = model.predict(test_X)
arc = abs((test_predictions-test_Y)/test_Y)
print("单位绝缘高度污闪电压最大误差为：",max(arc[:,0]))
print("单位爬电距离污闪电压最大误差为：",max(arc[:,1]))
print("单位绝缘高度污闪电压平均误差为：",np.mean(arc[:,0]))
print("单位爬电距离污闪电压平均误差为：",np.mean(arc[:,1]))

#保存模型
model.save('Adam_model.h5')
# #加载模型
# new_model = keras.models.load_model('Adam_model.h5')

#绘制loss曲线
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss [MSE]')
plt.plot(hist['epoch'], hist['mse'],label='mse')
plt.legend()
plt.show()

#绘制预测值与实际值的图像
number=np.array([1,2,3,4,5,6,7,8,9,10])
number.reshape(10,1)
plt.xlabel('sample')
plt.ylabel('Flashover voltage per unit insulation height')
plt.ylim([0,1.7])
plt.plot(number,test_Y[:,0])
plt.plot(number,test_predictions[:,0])
plt.scatter(number,test_Y[:,0],label='true value')
plt.scatter(number,test_predictions[:,0],label='predictive value')
plt.legend()
plt.show()

plt.xlabel('sample')
plt.ylabel('Flashover voltage per unit creepage distance')
plt.ylim([0,1.7])
plt.plot(number,test_Y[:,1])
plt.plot(number,test_predictions[:,1])
plt.scatter(number,test_Y[:,1],label='true value')
plt.scatter(number,test_predictions[:,1],label='predictive value')
plt.legend()
plt.show()
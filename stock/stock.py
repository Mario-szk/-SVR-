import numpy as np 

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 

#载入股指数据
import tushare as ts
df = ts.get_hist_data(code = '600848', start = '2017-1-1', end = '2018-1-1')
#保存在csv文件中
df.to_csv('600848.csv')


data = []
with open('600848.csv') as file:
    for line in file.readlines():
        line = line.strip().split(',')
        data.append(line)

#print(data)   
data = np.array(data)

#日成交最高价
high = data[1:, 2]
high = high.astype(float)

X = np.arange(169).reshape(-1,1)
y = high.tolist()
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=42)

clf = SVR(gamma='scale', C = 1.0, epsilon = 0.2)

clf.fit(X_train, y_train)

results = clf.predict(X_test)

print(results)

print(clf.score(X_test, y_test))

plt.figure()

#green真实值， red预测值
plt.plot(y_test, label = 'real-value', color = 'green')
plt.plot(results, label = 'prediction', color = 'red')

plt.show()
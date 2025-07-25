import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# 切换为TkAgg后端（需确保已安装tkinter库）
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 准备数据
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 特征（需为二维数组）
y = np.array([2, 4, 5, 4, 5])  # 标签

# 2. 训练模型
model = LinearRegression()
model.fit(x, y)

# 3. 输出参数
print(f"权重 w: {model.coef_[0]}")
print(f"偏置 b: {model.intercept_}")

# 4. 预测与评估
y_pred = model.predict(x)
mse = mean_squared_error(y, y_pred)
print(f"均方误差: {mse}")

# 5. 可视化
plt.scatter(x, y, color='blue', label='真实值')
plt.plot(x, y_pred, color='red', label='预测线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
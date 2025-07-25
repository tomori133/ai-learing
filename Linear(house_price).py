import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# 切换为TkAgg后端（需确保已安装tkinter库）
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing  # 替代波士顿数据集
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 加载数据（加州房价数据集，替代波士顿房价数据集）
housing = fetch_california_housing()
X = housing.data  # 特征：平均收入、房龄、平均房间数等
y = housing.target  # 目标：房价（单位：10万美元）

# 取单一特征（平均收入）做简单线性回归，便于可视化
X_single = X[:, [0]]  # 第0列：平均收入

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.3, random_state=42
)

# 3. 数据标准化（正则化必做步骤）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 定义模型（普通线性回归、Ridge、Lasso）
models = {
    "普通线性回归": LinearRegression(),
    "Ridge回归（L2正则化）": Ridge(alpha=1.0),  # alpha为正则化强度
    "Lasso回归（L1正则化）": Lasso(alpha=0.1)   # alpha为正则化强度
}

# 5. 训练与评估模型
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"mse": mse, "r2": r2, "model": model}
    print(f"{name} - 均方误差（MSE）: {mse:.4f}, 决定系数（R²）: {r2:.4f}")

# 6. 可视化结果（普通线性回归）
plt.figure(figsize=(10, 6))
# 绘制散点（真实值）
plt.scatter(X_test_scaled, y_test, color='blue', alpha=0.5, label='真实房价')
# 绘制回归线（预测值）
X_line = np.linspace(X_test_scaled.min(), X_test_scaled.max(), 100).reshape(-1, 1)
y_line = results["普通线性回归"]["model"].predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, label='回归预测线')

plt.xlabel('标准化后的平均收入')
plt.ylabel('房价（单位：10万美元）')
plt.title('平均收入与房价的线性回归关系')
plt.legend()
plt.show()

# 7. 对比不同正则化强度的影响（以Ridge为例）
alphas = [0.01, 0.1, 1, 10, 100]
ridge_mse = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    ridge_mse.append(mean_squared_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_mse, marker='o', color='green')
plt.xscale('log')  # 对数刻度，便于观察
plt.xlabel('正则化强度（alpha）')
plt.ylabel('测试集MSE')
plt.title('Ridge回归中正则化强度对MSE的影响')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 解决后端渲染问题
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 加载数据（取前两个特征便于可视化）
iris = datasets.load_iris()
X = iris.data[:, :2]  # 花萼长度和宽度（原始长度 150）
y = iris.target  # 标签（0,1,2三类，原始长度 150）

# 2. 同时筛选特征和标签（关键修正：基于原始y筛选，确保长度一致）
mask = y != 2  # 生成原始长度的布尔数组（150个元素）
X = X[mask]    # 筛选特征（长度变为 100）
y = y[mask]    # 筛选标签（长度变为 100）

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. 训练SVM模型（使用线性核）
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# 5. 预测与评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy}")

# 6. 可视化决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.title('SVM分类决策边界')
plt.show()
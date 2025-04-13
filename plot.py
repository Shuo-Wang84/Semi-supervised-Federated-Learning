import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建x和y数据的网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# 定义z数据 (例如，z = x^2 + y^2)
Z = X**2 + Y**2

# 创建3D绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 绘制曲面图
ax.plot_surface(X, Y, Z, cmap="Blues", edgecolor="none")

# 添加标签和网格
ax.set_xlabel("X轴")
ax.set_ylabel("Y轴")
ax.set_zlabel("Z轴")
ax.set_title("三维曲面图")

# 显示图形
plt.show()

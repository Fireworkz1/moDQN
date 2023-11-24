import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def showPareto(pareto_set):
    # 解压 solution_set 中的三个维度
    solution_set=[]
    for s in pareto_set:
        solution_set.append(s[0])
    x, y, z = zip(*solution_set)

    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, z, c='blue', marker='o', label='Pareto Front')

    # 设置坐标轴标签
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')

    # 显示图例
    ax.legend()

    # 显示图形
    plt.show()
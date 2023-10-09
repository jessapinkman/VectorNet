import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# shape = "398.38,824.38 397.26,817.01 394.25,811.72 389.34,808.52 382.55,807.40" / >
x = [398.38, 397.26, 394.25, 389.34, 382.55]
y = [824.38, 817.01, 811.72, 808.52, 807.40]
an = np.polyfit(x, y, 7)
p1 = np.poly1d(an)


yvals = np.polyval(an, x)      # 根据多项式系数计算拟合后的值

# 画图
plt.plot(x, y, '*', label='原数据点')
plt.plot(x, yvals, 'r', label='拟合后')
plt.xlabel('x 轴')
plt.ylabel('y 轴')
plt.legend(loc=4)               # 指定legend的位置
plt.title('曲线拟合')
# plt.show()

# x = [-1675, -1662, -1648, -1632, -1616, -1602, -1590, -1579, -1567, -1556, -1545, -1534,
#      -1525, -1515, -1506, -1496, -1485, -1473, -1462, -1450, -1439, -1428, -1417, -1405, -1399, -1389, -1379]
# y = [-3902, -3865, -3825, -3776, -3724, -3673, -3629, -3583, -3532, -3484, -3428, -3375,
#      -3326, -3269, -3221, -3167, -3114, -3050, -2990, -2927, -2868, -2808, -2750, -2684, -2653, -2600, -2547]
z1 = np.polyfit(x, y, 4)  # 用4次多项式拟合，输出系数从高到0
f1 = np.poly1d(z1)  # 使用次数合成多项式



def distance(x, y, x0, y0):
    """
    Return distance between point
    P[x0,y0] and a curve (x,y)
    """
    d_x = x - x0
    d_y = y - y0
    dis = np.sqrt(d_x ** 2 + d_y ** 2)
    return dis


def min_distance(x, y, P, precision=5):
    """
    Compute minimum/a distance/s between
    a point P[x0,y0] and a curve (x,y)
    rounded at `precision`.

    ARGS:
        x, y      (array)
        P         (tuple)
        precision (int)

    Returns min indexes and distances array.
    """
    # compute distance
    d = distance(x, y, P[0], P[1])
    d = np.round(d, precision)
    # find the minima
    glob_min_idxs = np.argwhere(d == np.min(d)).ravel()
    return glob_min_idxs, d


def f1_min_distance(P):
    x = np.linspace(0, 1000, 1000)
    y = f1(x)

    min_idxs, dis = min_distance(x, y, P)
    return min(dis)


if __name__ == '__main__':
    # P = (394, 815)
    # d = f1_min_distance(P)
    # plt.plot(394, 815, 'bo')
    # print(d)
    # plt.show()
    # traj_directory = "/home/pinkman/PycharmProjects/VectorNet/logsim/processed_trajectory_train"
    # for filename in sorted(os.listdir(traj_directory)):
    #     if filename.endswith(".csv"):
    #         file_path = os.path.join(traj_directory, filename)
    #         traj_df = pd.read_csv(file_path)
    #         traj_df = traj_df.dropna(subset=['lane_id'])
    #         traj_df.to_csv(file_path)
    #         print(filename)
    #         # process trajectory data
    import os

    dir_path = '//data_process/trajectory/train_1.2'
    index = 0

    for i in range(3901):
        # 生成新的文件名
        old_filename = str(i) + ".csv"
        new_filename = str(index) + ".csv"
        # 构建旧文件路径和新文件路径
        old_filepath = os.path.join(dir_path, old_filename)
        if os.path.exists(old_filepath):
            new_filepath = os.path.join(dir_path, new_filename)
            os.rename(old_filepath, new_filepath)
            index += 1
        # 重命名文件
        # os.rename(old_filepath, new_filepath)





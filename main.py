import numpy as np
from matplotlib import pyplot as plt

# 两点之间距离
def distance(points):
    n = len(points)
    dist = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            dist[i,j] = np.sqrt((points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2)
    return dist

# 路径长度
def path_distance(path):
    dis = 0
    for i in range(len(path)-1):
        dis += dist[path[i]][path[i+1]]
    return dis

# du
def diff_U(V, dist):
    a = np.sum(V, axis=0) - 1
    b = np.sum(V, axis=1) - 1
    t1 = np.zeros((N, N))
    t2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            t1[i, j] = a[j]
    for i in range(N):
        for j in range(N):
            t2[j, i] = b[j]
    c_1 = V[:, 1:N]
    c_0 = np.zeros((N, 1))
    c_0[:, 0] = V[:, 0]
    c = np.concatenate((c_1, c_0), axis=1)
    c = np.dot(dist, c)
    return -A * (t1 + t2) - D * c

# 更新
def update_U(U, diff_U, step):
    update_U = U + diff_U * step
    return update_U

def update_V(U, U0):
    return 0.5 * (1 + np.tanh(U / U0))

# energy_function
def energy_func(V, dist):
    t1 = np.sum(np.power(np.sum(V, axis=0) - 1, 2))
    t2 = np.sum(np.power(np.sum(V, axis=1) - 1, 2))
    idx = [i for i in range(1, N)]
    idx = idx + [0]
    Vt = V[:, idx]
    t3 = dist * Vt
    t3 = np.sum(np.sum(np.multiply(V, t3)))
    e = 0.5 * (A * (t1 + t2) + D * t3)
    return e

def check_path(V):
    newV = np.zeros([N, N])
    route = []
    for i in range(N):
        mm = np.max(V[:, i])
        for j in range(N):
            if V[j, i] == mm:
                newV[j, i] = 1
                route += [j]
                break
    return route, newV

# 可视化画出哈密顿回路和能量趋势
def plot_route(points, path, energys):
    fig = plt.figure()
    # 绘制哈密顿回路
    ax1 = fig.add_subplot(121)
    plt.xlim(0, np.max(points[:,0]+1))
    plt.ylim(0, np.max(points[:,1]+1))
    for (start, end) in path:
        p1 = plt.Circle(points[start], 0.2, color='red')
        p2 = plt.Circle(points[end], 0.2, color='red')
        ax1.add_patch(p1)
        ax1.add_patch(p2)
        ax1.plot((points[start][0], points[end][0]), (points[start][1], points[end][1]), color='red')
        ax1.annotate(text=chr(97 + end), xy=points[end], xytext=(-8, -4), textcoords='offset points', fontsize=20)
    ax1.axis('equal')
    ax1.grid()
    # 绘制能量趋势图
    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(0, len(energys), 1), energys, color='red')
    plt.show()


if __name__ == '__main__':
    points = np.array([[3, 3], [2, 7], [1, 5], [4, 6], [4, 4], [4, 2], [7, 3], [5, 4]])
    dist = distance(points)
    print(dist)
    N = len(points)
    # 设置初始值
    A = N * N
    D = N / 2
    U0 = 0.0009
    step = 0.0001
    num_iter = 10000
    # 初始化
    U = 1 / 2 * U0 * np.log(N - 1) + (2 * (np.random.random((N, N))) - 1)
    V = update_V(U, U0)
    energys = np.array([0.0 for x in range(num_iter)])
    best_distance = np.inf
    best_route = []  # 最优路线
    H_path = []
    for n in range(num_iter):
        du = diff_U(V, dist)
        U = update_U(U, du, step)
        V = update_V(U, U0)
        energys[n] = energy_func(V, dist)
        route, newV = check_path(V)
        if len(np.unique(route)) == N:
            route.append(route[0])
            dis = path_distance(route)
            if dis < best_distance:
                H_path = []
                best_distance = dis
                best_route = route
                [H_path.append((route[i], route[i + 1])) for i in range(len(route) - 1)]
                print('第{}次迭代找到的次优解距离为：{}，能量为：{}，路径为：'.format(n, best_distance, energys[n]))
                [print(chr(97 + v), end=',' if i < len(best_route) - 1 else '\n') for i, v in enumerate(best_route)]
    if len(H_path) > 0:
        plot_route(points, H_path, energys)
    else:
        print('没有找到最优解')

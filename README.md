# TSP_Hopfield

##### Hopfield神经网络求解TSP
##### 1 初始化权值（A,D,U0）
##### 2 计算n个城市的距离矩阵: dist[x,y]
##### 3 初始化神经网络: 输入: U[x,i], 输出: V[x,i]
##### 4 利用动力微分方程计算: dUxi/dt
##### 5 由一阶欧拉方法更新计算: Uxi(t+1) = Uxi(t) + dUxi/dt * step
##### 6 由非线性函数sigmoid更新计算: Vxi(t) = 0.5 * (1 + tanh(Uxi/U0))
##### 7 计算能量函数 E
##### 8 检查路径是否合法

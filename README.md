# TSP_Hopfield

##### Hopfield神经网络求解TSP
##### 1 初始化权值
##### 2 计算n个点的距离矩阵: distance[x,y]
##### 3 初始化神经网络: 输入: U[x,i], 输出: V[x,i]
##### 4 利用微分方程计算U的微分: dUxi/dt
##### 5 更新计算U: Uxi(t+1) = Uxi(t) + dUxi/dt * step
##### 6 由非线性函数更新计算V: Vxi(t) = 0.5 * (1 + tanh(Uxi/U0))
##### 7 计算能量函数 E
##### 8 检查路径是否合法

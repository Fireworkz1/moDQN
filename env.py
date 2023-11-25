ContainerNumber = 6  # 容器数
NodeNumber = 5  # 节点数
ServiceNumber = 4  # 服务种类
ResourceType = 3  # 资源类型数,CPU,Mem,BandWidth
service_containernum = [1, 1, 3, 1]  # 每种服务需要启动的容器数量（按编号索引）
service_container = [[0], [1], [2, 3, 4], [5]]  # 每种服务对应的容器编号（按编号索引）
service_container_relationship = [0, 1, 2, 2, 2, 3]  # 上面两个数组的对应
node_delay = [100, 200, 200, 100, 150]  # 每个节点的延迟
node_loss = [23, 12, 8, 15, 8]  # 每个节点的丢包%
alpha = 0.5  # reward weighting factor
beta = [0.333, 0.333, 0.333]
count = 0
CPUnum = 4
Mem = CPUnum * 1024
BandWidth = 3
e_greed = 0.1 # 模型学习率
e_greed_decrement = 1e-6

import numpy as np


class Env():
    def __init__(self):
        # State

        self.State = []
        self.node_state_queue = []
        self.container_state_queue = []
        self.action_queue = []
        self.loss_state_query = []
        self.delay_state_query = []
        self.prepare()

    def prepare(self):

        self.container_state_queue = [-1, 0.5 / CPUnum, 128 / Mem, 2 / BandWidth,
                                      -1, 0.5 / CPUnum, 256 / Mem, 2 / BandWidth,
                                      -1, 0.5 / CPUnum, 256 / Mem, 2 / BandWidth,
                                      -1, 0.5 / CPUnum, 256 / Mem, 1 / BandWidth,
                                      -1, 0.5 / CPUnum, 256 / Mem, 2 / BandWidth,
                                      -1, 0.5 / CPUnum, 128 / Mem, 1 / BandWidth]  # 设置硬件条件

        for i in range(NodeNumber):
            # self.node_state_queue.extend([0, 0, 0, 0, 0, 0, 0, 0])
            self.node_state_queue.extend([0,0,0,0,0,0,0,0,0])  # 微服务启动容器数+3
        self.loss_state_query = [0,0,0,0,0,0]
        self.delay_state_query = [0,0,0,0,0,0]
        self.State = self.container_state_queue + self.node_state_queue + self.loss_state_query + self.delay_state_query
        self.action = [-1, -1]
        self.action_queue = [-1, -1]

        # Communication weight between microservices
        # 四种微服务间的通信成本
        self.service_weight = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 2], [0, 0, 2, 0]]
        # Communication distance between nodes
        # 节点间的（物理）通信距离
        self.Dist = [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]

    def ContainerCost(self, i, j):
        # to calculate the distance between container i and j
        m = -1
        n = -1
        m = self.container_state_queue[i * (ResourceType + 1)]
        n = self.container_state_queue[j * (ResourceType + 1)]

        p = service_container_relationship[i]
        q = service_container_relationship[j]

        if self.Dist[m][n] != 0 and (p != q):
            container_dist = self.Dist[m][n]
        else:
            container_dist = 0
        return container_dist

    def CalcuCost(self, i, j):
        # to calculate the communication cost between container i and j
        cost = 0
        interaction = self.service_weight[i][j] / (service_containernum[i] * service_containernum[j])
        for k in range(len(service_container[i])):
            for l in range(len(service_container[j])):
                cost += self.ContainerCost(service_container[i][k], service_container[j][l]) * interaction
        return cost

    def sumCost(self):
        Cost = 0
        for i in range(ServiceNumber):
            for j in range(ServiceNumber):
                Cost += self.CalcuCost(i, j)
        return 0.5 * Cost

    def CalcuVar(self):
        NodeCPU = []
        NodeMemory = []
        NodeBandWith = []
        Var = 0
        for i in range(NodeNumber):
            U = self.node_state_queue[i * (ContainerNumber + 3) + ContainerNumber]
            M = self.node_state_queue[i * (ContainerNumber + 3) + (ContainerNumber + 1)]
            B = self.node_state_queue[i * (ContainerNumber + 3) + (ContainerNumber + 2)]
            NodeCPU.append(U)
            NodeMemory.append(M)
            NodeBandWith.append(B)
            if NodeCPU[i] > 1 or NodeMemory[i] > 1 or NodeBandWith[i] > BandWidth:
                Var -= 10
                # Variance of node load
        Var += beta[0] * np.var(NodeCPU) + beta[1] * np.var(NodeMemory) + beta[2] * np.var(NodeBandWith)
        return Var

    def CalcuCostFin(self):
        re = 0
        g1 = self.sumCost()
        g1 = g1 / 4
        g2 = self.CalcuVar()
        g2 = g2 / 0.052812500000000005
        re += alpha * g1 + (1 - alpha) * g2
        return 100 * re

    def state_update(self, container_state_queue, node_state_queue, loss_state_query, delay_state_query):
        self.State = container_state_queue + node_state_queue + loss_state_query + delay_state_query

    def update(self):
        # update state
        if self.action[0] >= 0 and self.action[1] >= 0:
            # update container state
            self.container_state_queue[self.action[1] * (ResourceType + 1)] = self.action[0]
            # update node state
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + self.action[1]] = 1
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + ContainerNumber] += self.container_state_queue[self.action[1] * (ResourceType + 1) + 1]
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + (ContainerNumber + 1)] += self.container_state_queue[self.action[1] * (ResourceType + 1) + 2]
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + (ContainerNumber + 2)] += self.container_state_queue[self.action[1] * (ResourceType + 1) + 3]
            self.loss_state_query[self.action[1]] = node_loss[self.action[0]]
            self.delay_state_query[self.action[1]] = node_delay[self.action[0]]
            self.action_queue.append(self.action)
        else:
            print("invalid action")
            self.node_state_queue = []
            self.container_state_queue = []
            self.delay_state_query = []
            self.loss_state_query = []
            self.action_queue = []

            self.prepare()
        self.state_update(self.container_state_queue, self.node_state_queue, self.loss_state_query, self.delay_state_query)
        return self.State

    # def CalcuLoss(self):
    #     loss = node_loss[self.action[0]]
    #     feature = (40 - loss) * 12
    #     return feature

    # def CalcuDelay(self):
    #     # act[0]为部署在几号节点上，act[1]为部署第几个微服务容器实例
    #     delay = node_delay[self.action[0]]
    #     feature = (500 - delay) / 10 / ContainerNumber
    #     return feature

    def step(self, action):
        # input: action(Targetnode，ContainerIndex)
        # output: next state, cost, done
        global count
        self.action = self.index_to_act(action)
        self.update()

        feature1 = self.CalcuCostFin()  # 35左右
        feature2 = sum(self.loss_state_query)   # 35/6左右
        feature3 = sum(self.delay_state_query)     # 35/6左右

        done = False
        count = 0

        # 判断当前为第几步,用来判断是否完成迭代
        for i in range(ContainerNumber):
            if self.container_state_queue[(ResourceType + 1) * i] != -1:
                count += 1
        if count == ContainerNumber:
            done = True

        # 返回当前的状态、通信开销以及资源方差、loss和、延迟和
        return self.State, feature1, feature2, feature3, done

    def reset(self):
        self.node_state_queue = []
        self.container_state_queue = []
        self.prepare()
        return self.State, self.action

    def index_to_act(self, index):
        act = [-1, -1]
        act[0] = int(index / ContainerNumber)
        act[1] = index % ContainerNumber
        return act

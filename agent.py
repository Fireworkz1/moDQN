#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import numpy as np
import paddle
import parl


from env import ContainerNumber
from env import NodeNumber
from policy import getQnetwork
from  policy import  merge_q

flag = []
flag_temp = []
for o in range(ContainerNumber * NodeNumber):
    flag.append(0)
    flag_temp.append(0)


class Agent(parl.Agent):
    def __init__(self, algorithm_1, algorithm_2, algorithm_3, act_dim, e_greed=0.1, e_greed_decrement=0):
        super(Agent, self).__init__(algorithm_1)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim
        self.alg_2 = algorithm_2
        self.alg_3 = algorithm_3

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        sample = np.random.random()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
            while flag[act] == 1 or flag_temp[act % ContainerNumber] == 1:
                act = np.random.randint(0, self.act_dim)
            flag[act] = 1
            flag_temp[act] = 1
            flag_temp[act % ContainerNumber] = 1
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act



    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        global flag
        global flag_temp
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q1 = self.alg.predict(obs)
        pred_q2 = self.alg_2.predict(obs)
        pred_q3 = self.alg_3.predict(obs)

        act,flag,flag_temp=getQnetwork(1,pred_q1,pred_q2,pred_q3,act_dim=self.act_dim,flag=flag,flag_temp=flag_temp)

        # 如果后面要修改多目标q函数从这里改

        # 下面这个函数返回的是采用不同act时对应的act在reward上的排名值，从小到大
        # 比如，[3.2, 1.5, 4.1, 2.7 ]调用后为[1,3,0,2]

        return act

    def learn(self, obs, act, reward1, reward2, reward3, next_obs, terminal):
        """ 根据训练数据更新一次模型参数
        """
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
            self.alg_2.sync_target()
            self.alg_3.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward1 = np.expand_dims(reward1, axis=-1)
        reward2 = np.expand_dims(reward2, axis=-1)
        reward3 = np.expand_dims(reward3, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward1 = paddle.to_tensor(reward1, dtype='float32')
        reward2 = paddle.to_tensor(reward2, dtype='float32')
        reward3 = paddle.to_tensor(reward3, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        loss_1 = self.alg.learn(obs, act, reward1, next_obs, terminal)  # 训练一次网络
        loss_2 = self.alg_2.learn(obs, act, reward2, next_obs, terminal)
        loss_3 = self.alg_3.learn(obs, act, reward3, next_obs, terminal)
        return float(loss_1), float(loss_2), float(loss_3)

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

import copy

import paddle
import parl

from weight import getWeightReward
from weight import getWeightQfunc
from weight import getWeightLoss


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            gamma (float): reward的衰减因子
            lr (float): learning_rate，学习率.
        """
        # checks

        """
        隐含定义调用关系如下
        model = Model(obs_dim=obs_shape, act_dim=action_dim)
    algorithm = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        act_dim=action_dim,
        e_greed=e_greed,  # 有一定概率随机选取动作，探索
        e_greed_decrement=e_greed_decrement)  # type: ignore # 随着训练逐步收敛，探索的程度慢慢降低
        """
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')  # type: ignore
        self.optimizer1 = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.head1.parameters())  # 使用Adam优化器优化第一个目标
        self.optimizer2 = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.head2.parameters())  # 使用Adam优化器优化第二个目标
        self.optimizer3 = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.head3.parameters())  # 使用Adam优化器优化第三个目标

    def predict(self, obs):
        """ 使用self.model的网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model(obs)

    def learn(self, obs, action, reward1, reward2, reward3, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 获取Q预测值
        # 这里的pred_values为一个元组了
        pred_values = self.model(obs)

        # 如果模型输出是一个元组，可以通过拆包获取每个头的输出

        pred_values1 = pred_values[0]
        pred_values2 = pred_values[1]
        pred_values3 = pred_values[2]

        action_dim = pred_values1.shape[-1]
        action = paddle.squeeze(action, axis=-1)
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = paddle.nn.functional.one_hot(
            action, num_classes=action_dim)
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        pred_value1 = pred_values1 * action_onehot
        pred_value2 = pred_values2 * action_onehot
        pred_value3 = pred_values3 * action_onehot
        #  ==> pred_value = [[3.9]]
        pred_value1 = paddle.sum(pred_value1, axis=1, keepdim=True)
        pred_value2 = paddle.sum(pred_value2, axis=1, keepdim=True)
        pred_value3 = paddle.sum(pred_value3, axis=1, keepdim=True)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        with paddle.no_grad():
            max_v1 = self.target_model(next_obs)[0].max(1, keepdim=True)
            target1 = reward1 + (1 - terminal) * self.gamma * max_v1
        loss1 = self.mse_loss(pred_value1, target1)
        with paddle.no_grad():
            max_v2 = self.target_model(next_obs)[1].max(1, keepdim=True)
            target2 = reward2 + (1 - terminal) * self.gamma * max_v2
        loss2 = self.mse_loss(pred_value2, target2)
        with paddle.no_grad():
            max_v3 = self.target_model(next_obs)[2].max(1, keepdim=True)
            target3 = reward3 + (1 - terminal) * self.gamma * max_v3
        loss3 = self.mse_loss(pred_value3, target3)

        # 计算每个 Q(s,a) 与 target_Q的均方差，得到loss
        # 训练头部1
        self.optimizer1.clear_grad()
        loss1.backward(retain_graph=True)
        self.optimizer1.step()

        # 训练头部2
        self.optimizer2.clear_grad()
        loss2.backward(retain_graph=True)
        self.optimizer2.step()

        # 训练头部3
        self.optimizer3.clear_grad()
        loss3.backward()
        self.optimizer3.step()


        w=getWeightLoss()
        loss = w[0] * loss1 + w[1] * loss2 + w[2] * loss3
        loss = loss / sum(w)

        return loss, loss1, loss2, loss3

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)

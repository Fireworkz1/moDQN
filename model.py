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

import paddle.nn as nn
import paddle.nn.functional as F
import parl



class Model(parl.Model):
    """ 使用全连接网络.
    """

    def __init__(self, input_dim, hid_1, hid_2, act_dim):
        super(Model, self).__init__()
        hid1_size = hid_1
        hid2_size = hid_2
        # 3层全连接层
        # 三头网络
        self.fc1 = nn.Linear(input_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)    
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        return Q



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

import logging

from agent import Agent
from agent import flag
from agent import flag_temp
from algorithm import DQN  # from parl.algorithms import DQN  # parl >= 1.3.1
# 添加部分全局变量
from env import ContainerNumber
from env import Env
from env import NodeNumber
from env import ResourceType
from env import e_greed
from env import e_greed_decrement
from model import Model
from replay_memory import ReplayMemory
from reward import Culreward
from weight import getWeightReward

# 检查版本

LEARN_FREQ = 6  # learning frequency
MEMORY_SIZE = 10000  # size of replay memory
MEMORY_WARMUP_SIZE = 2000
BATCH_SIZE = 30
LEARNING_RATE = 0.001
GAMMA = 0.9

sc_comm = 0
sc_var = 0
flag1 = 1
ep = 0
round = 0
# allCost = [[], [], [], [], [], []]
allReward = [[], [], [], [], [], []]
rewardavg=0

# ------------
# LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
# MEMORY_SIZE = 200000  # replay memory的大小，越大越占用内存
# MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
# BATCH_SIZE = 64  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
# LEARNING_RATE = 0.0005  # 学习率
# GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
# 训练一个episode
# def run_train_episode(agent, env, rpm):
#     total_reward = 0
#     obs = env.reset()
#     step = 0
#     while True:
#         step += 1
#         action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
#         next_obs, reward, done, _ = env.step(action)
#         rpm.append((obs, action, reward, next_obs, done))

#         # train model
#         if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
#             # s,a,r,s',done
#             (batch_obs, batch_action, batch_reward, batch_next_obs,
#              batch_done) = rpm.sample(BATCH_SIZE)
#             train_loss = agent.learn(batch_obs, batch_action, batch_reward,
#                                      batch_next_obs, batch_done)

#         total_reward += reward
#         obs = next_obs
#         if done:
#             break
#     return total_reward

def run_train_episode(agent, env, rpm):
    global flag1
    # allCost =     [[], [], [], [], [], []]
    global allReward  # allReward = [[], [], [], [], [], []]
    global ep
    global rewardavg
    # ------------

    global round

    obs_list = []
    next_obslist = []
    action_list = []
    done_list = []

    ep += 1
    obs, action = env.reset()

    step = 0

    for o in range(ContainerNumber * NodeNumber):
        flag_temp[o] = 0
        flag[o] = 0
    flag1 -= 1
    while True:

        step += 1

        # 选择一种动作（随机或最优）
        # act[0]为节点号
        # act[1]为容器编号
        action = agent.sample(obs)
        # 与环境交互
        # container_state_queue中的-1变为该容器部署的节点号（nextobs中）
        # node_state_value中每8号代表一个节点，前六位为容器是否部署在该node（部署为1），后两位为节点的资源占用情况
        next_obs, feature1, feature2, feature3, done = env.step(action)

        # 记录当前episode的数据
        obs_list.append(obs)
        action_list.append(action)
        next_obslist.append(next_obs)
        done_list.append(done)

        # ------------
        reward1 = 0
        reward2 = 0
        reward3 = 0

        if step == 6:

            # feature1：35左右
            # feature2：35左右
            # feature3：35左右
            reward1, reward2, reward3 = Culreward(feature1, feature2, feature3)

            for i in range(6):
                rpm.append(
                    (obs_list[i], action_list[i], reward1, reward2, reward3, next_obslist[i], done_list[i]))
            print(action_list)

            # 输出到日志
            rewardsum = reward1 + reward3 + reward2
            rewardavg=(rewardsum+rewardavg*(ep-1))/ep

            root_logger = logging.getLogger()
            for h in root_logger.handlers[:]:
                root_logger.removeHandler(h)
            logging.basicConfig(level=logging.INFO, filename='feature-reward.log')
            logging.info(
                'episode:{} round:{} Ravg:{:.2f} Reward1:{:.2f} Reward2:{:.2f} Reward3:{:.2f} Feature1:{:.2f} Feature2:{:.2f} Feature3:{:.2f}'.format(
                    ep, round, rewardavg, reward1, reward2, reward3, feature1, feature2, feature3))

        # 如果rpm池已满，开始训练
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward1, batch_reward2, batch_reward3, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)

            loss1, loss2, loss3 = agent.learn(batch_obs, batch_action, batch_reward1, batch_reward2,
                                              batch_reward3,
                                              batch_next_obs,
                                              batch_done)  # s,a,r,s',done
            with open("trainloss.txt", "a") as f:
                f.write("ep:%d,loss1:%.3f,loss2:%.3f,loss3:%.3f \n" % (ep, loss1, loss2, loss3))

        obs = next_obs
        if done:
            break

    return feature1, feature2, feature3, reward1, reward2, reward3, action_list


def calPareto(f1, f2, f3):
    return


def main():
    global sc_comm, sc_var
    global rewardavg
    env = Env()
    obs_shape = ContainerNumber * (ResourceType + 1) + NodeNumber * (
            ContainerNumber + 3) + ContainerNumber * 2  # *3对应containerstate数组，每个container三个值；后半对应nodestate数组
    action_dim = ContainerNumber * NodeNumber

    rpm = ReplayMemory(MEMORY_SIZE)  # Target1的经验回放池

    # 根据parl框架构建agent
    model_1 = Model(obs_shape, 128, 128, action_dim)
    model_2 = Model(obs_shape, 128, 128, action_dim)
    model_3 = Model(obs_shape, 128, 128, action_dim)
    algorithm_1 = DQN(model_1, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_2 = DQN(model_2, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_3 = DQN(model_3, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm_1, algorithm_2, algorithm_3,
        act_dim=action_dim,
        e_greed=e_greed,  # 有一定概率随机选取动作，探索
        e_greed_decrement=e_greed_decrement)  # type: ignore # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        # MEMORY_WARMUP_SIZE=2000
        run_train_episode(agent, env, rpm)

    max_episode = 3000
    pareto_set=[]
    # start train
    global round

    while round < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part

        with open("cost.txt", "a") as f:
            f.write("开始训练 round:" + str(round) + "\n")

        f1, f2, f3, r1, r2, r3,action_list = run_train_episode(agent, env, rpm)
        inPareto=calPareto(f1,f2,f3)
        if inPareto:
            actset=[]
            for i in action_list:
                act = [-1, -1]
                act[0] = int(i / ContainerNumber)
                act[1] = i % ContainerNumber
                actset.append(act)
            pareto_set.append(actset)
        round += 1
        print("ep,round:" + str(ep) + " " + str(round))

        if round>(max_episode-5):
            with open("all_trains.txt", "a") as f:
                f.write("final ravg" + str(rewardavg)+str(action_list) + "\n")

    # 观察模型是否趋于较好结果

    # 训练结束，保存模型
    save_path = './mdqn_model.ckpt'
    agent.save(save_path)



if __name__ == '__main__':
    main()

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
# allCost = [[], [], [], [], [], []]
allReward = [[], [], [], [], [], []]

test_evareward = 0
# ------------
min_feature1 = 100000000000
min_feature2 = 100000000000
min_feature3 = 100000000000


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

    # ------------
    global min_feature1
    global min_feature2
    global min_feature3

    obs_list = []
    next_obslist = []
    action_list = []
    done_list = []

    total_reward1 = 0
    total_reward2 = 0
    total_reward3 = 0
    ep += 1
    obs, action = env.reset()

    step = 0
    # minc ost
    minReward = -1

    for o in range(ContainerNumber * NodeNumber):
        flag_temp[o] = 0
        flag[o] = 0
    flag1 -= 1
    # ------------
    feature1_temp = 0
    feature2_temp = 0
    feature3_temp = 0

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
        feature1_temp = feature1  # feature1（cost&var）为即时型
        feature2_temp += feature2  # feature2（通信延迟）为累加型
        feature3_temp += feature3  # feature3（丢包率）为平均型

        reward1 = 0
        reward2 = 0
        reward3 = 0
        if flag1 == 0:
            if step == 6:
                # 如果是第一个episode，直接存入
                # 如果feature不好，则舍弃这一个episode，下一个episode看作第一个episode
                if feature1_temp > 0 and feature2_temp > 0 and feature3_temp > 0:
                    min_feature1 = feature1_temp
                    min_feature2 += feature2_temp
                    min_feature3 += feature3_temp

                else:
                    flag1 += 1
                    min_feature1 = 100000000000
                    min_feature2 = 100000000000
                    min_feature3 = 100000000000
                    break

        # 如果不是第一个episode，进入reward环节
        else:
            if step == 6:
                # feature1：35左右
                # feature2：35左右
                # feature3：35左右
                feature2 = feature2_temp
                feature3 = feature3_temp / ContainerNumber
                reward1, reward2, reward3, min_feature1, min_feature2, min_feature3 = Culreward(feature1, feature2,
                                                                                                feature3, min_feature1,
                                                                                                min_feature2,
                                                                                                min_feature3)

                for i in range(6):
                    rpm.append(
                        (obs_list[i], action_list[i], reward1, reward2, reward3, next_obslist[i], done_list[i]))
        w = getWeightReward()
        rewardAvg = reward1 * w[0] + reward2 * w[1] + reward3 * w[2]
        rewardAvg /= sum(w)
        # 输出到日志
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        logging.basicConfig(level=logging.INFO, filename='details.log')
        logging.info(
            'episode:{} step:{} Reward1:{} Reward2:{} Reward3:{} Reward:{} Action:{}'.format(
                ep, step, reward1, reward2, reward3, rewardAvg, env.index_to_act(action)))

        # 如果rpm池已满，开始训练
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward1, batch_reward2, batch_reward3, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)

            train_loss, loss1, loss2, loss3 = agent.learn(batch_obs, batch_action, batch_reward1, batch_reward2,
                                                          batch_reward3,
                                                          batch_next_obs,
                                                          batch_done)  # s,a,r,s',done
            with open("trainloss.txt", "a") as f:
                f.write("ep:%d,loss:%.3f,loss1:%.3f,loss2:%.3f,loss3:%.3f \n" % (ep, train_loss, loss1, loss2, loss3))
        obs = next_obs
        if done:
            break
    w = getWeightReward()
    featureAvg = feature1 * w[0] + feature2 * w[1] + feature3 * w[2]
    featureAvg /= sum(w)
    rewardAvg = reward1 * w[0] + reward2 * w[1] + reward3 * w[2]
    rewardAvg /= sum(w)
    return feature1, feature2, feature3, featureAvg, reward1, reward2, reward3, rewardAvg


def main():
    global sc_comm, sc_var
    env = Env()
    obs_shape = ContainerNumber * (ResourceType + 1) + NodeNumber * (
            ContainerNumber + 3)  # *3对应containerstate数组，每个container三个值；后半对应nodestate数组
    action_dim = ContainerNumber * NodeNumber

    rpm = ReplayMemory(MEMORY_SIZE)  # Target1的经验回放池

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_shape, act_dim=action_dim)
    algorithm = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
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

    max_episode = 2000

    # start train
    round = 0

    while round < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        f1_list = []
        f2_list = []
        f3_list = []
        favg_list = []
        r1_list = []
        r2_list = []
        r3_list = []
        ravg_list = []
        with open("cost.txt", "a") as f:
            f.write("开始训练 round:" + str(round) + "\n")
        for i in range(50):
            f1, f2, f3, favg, r1, r2, r3, ravg = run_train_episode(agent, env, rpm)
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            favg_list.append(favg)
            r1_list.append(r1)
            r2_list.append(r2)
            r3_list.append(r3)
            ravg_list.append(ravg)
            round += 1
            print("ep,round:"+str(ep)+" "+str(round))
            if (round < 50):
                with open("cost.txt", "a") as f:
                    f.write("round=%d,f1=%.6f,f2=%.6f,f3=%.6f,favg=%.6f,r1=%.6f,r2=%.6f,r3=%.6f,ravg=%.6f \n" % (
                    round, f1, f2, f3, favg, r1, r2, r3, ravg))

        # test part       render=True 查看显示效果
        # eval_reward = run_evaluate_episodes(agent, env, render=False)
        # logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(
        #     episode, agent.e_greed, eval_reward))

        with open("cost.txt", "a") as f:
            f.write("round=%d,f1=%.6f,f2=%.6f,f3=%.6f,favg=%.6f,r1=%.6f,r2=%.6f,r3=%.6f,ravg=%.6f \n" % (
                round, sum(f1_list) / len(f1_list), sum(f2_list) / len(f2_list), sum(f3_list) / len(f3_list),
                sum(favg_list) / len(favg_list),
                sum(r1_list) / len(r1_list),
                sum(r2_list) / len(r2_list),
                sum(r3_list) / len(r3_list), sum(ravg_list) / len(ravg_list)))
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        logging.basicConfig(level=logging.INFO, filename='a.log')
        logging.info('round:{} e_greed:{} FeatureAvg: {} RewardAvg:{} Action:{}'.format(
            round, agent.e_greed, sum(favg_list) / len(favg_list), sum(ravg_list) / len(ravg_list), env.action_queue))

    # 训练结束，保存模型
    save_path = './mdqn_model.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    main()

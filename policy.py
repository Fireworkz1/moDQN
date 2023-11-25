import numpy as np
import random
from env import ContainerNumber
from  env import NodeNumber
policy_=""
from pareto import QPareto
def merge_q(q1, q2, q3, act_dim, flag, flag_temp):

    wq = [1 / 3, 1 / 3, 1 / 3]
    merged_q = q1 * wq[0] + q2 * wq[1] + q3 * wq[2]
    # q_index = np.argsort(merged_q)
    # act为preq中的最大值原先对应的索引,原来的列表里第index[i]位 是第i小的数(较大的)
    # act = q_index[act_dim - 1]
    # q_index = np.argsort(merged_q)
    q_index = np.argsort(merged_q)
    # act为preq中的最大值原先对应的索引,原来的列表里第index[i]位 是第i小的数(较大的)
    act = q_index[act_dim - 1]
    i = 1
    while flag[act] == 1 or flag_temp[act % ContainerNumber] == 1:
        i += 1
        act = q_index[act_dim - i]  # 选择Q最大的下标，即对应的动作
    flag[act] = 1
    flag_temp[act] = 1
    flag_temp[act % ContainerNumber] = 1

    return act, flag, flag_temp


def pareto_q(q1, q2, q3, act_dim, flag, flag_temp):
    solution_set=[]
    for i in range(act_dim):
        if flag[i]==0 and flag_temp[i% ContainerNumber]==0:
            solution_set.append([[q1[i],q2[i],q3[i]],i])
    pareto_set=QPareto(solution_set)

    act = random.choice(pareto_set)[1]

    flag[act] = 1
    flag_temp[act] = 1
    flag_temp[act % ContainerNumber] = 1
    return act, flag, flag_temp


def getQnetwork(policy, q1, q2, q3, act_dim, flag, flag_temp):
    global policy_
    if policy == 0:
        policy_="weighted policy"
        act, flag, flag_temp = merge_q(q1, q2, q3, act_dim, flag, flag_temp)
    elif policy == 1:
        policy_="pareto policy"
        act, flag, flag_temp = pareto_q(q1, q2, q3, act_dim, flag, flag_temp)
    else:
        raise ValueError("invalid policy number.")

    return act, flag, flag_temp

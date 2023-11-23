# moDQN

查看训练效果：feature-reward.log
记录每次完整训练的最后五次迭代： all_trains.txt
reward算法改为归一化
调参入口：
train.py calPareto(f1, f2, f3):计算f1，f2，f3是否为在pareto以内
weight.py getWeightQfunC():当前训练Q网络合并的权重
reward.py 设置reward

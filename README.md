# moDQN
   
<span style="color: red; font-size: 18px;">若要修改env，一定要先删除pareto_merge.txt，pareto_final.txt，pareto_set.txt！ </span>  
<span style="color: blue; font-size: 16px;">start.py:程序总入口，可多次执行主程序，然后执行pareto_sum脚本统计多次执行的最优解集  </span>

reward算法为归一化  

feature-reward.log：查看训练效果  
all_trains.txt：记录每次完整训练的最后五次迭代
pareto_set.txt：最优解集（可用pareto_sum.py合并）  
pareto_details.txt：最优解集迭代过程：  
pareto_merge.txt：多次学习后合并pareto，可以观察模型是否是否合理：  
pareto_final.txt:保存全部最优解  

强化学习基本组件：  
env.py  
train.py  
agent.py  
algorithm.py  
model.py  
replay_memory.py  
自定义文件：  
policy.py：设置选择Q网络的策略
reward.py：设置reward  
pareto.py:计算pareto前沿相关  
show.py:画图  
delete_script.py:删除log等  

单次运行以外：  
pareto_sum.py：将保存在pareto_set.txt中的多次最优解集合并  
start.py:程序总入口，可多次执行主程序，然后执行pareto_sum脚本统计多次执行的最优解集  


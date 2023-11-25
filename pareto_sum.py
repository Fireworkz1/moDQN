from pareto import mergePareto
import os
from show import showPareto
from env import CPUnum
from env import Mem
from env import BandWidth
from policy import policy_
times = 1
temp = 0


def parse_line(line):
    global temp
    global times
    # 将一行解析为 [round, act, feature] 的列表
    # 移除首尾的空格，并分割为多个部分
    parts = line.strip().split(';')

    # 提取 round 部分
    round_part = parts[0].split(':')[1].strip()
    round_num = int(round_part)
    if round_num < temp:
        times += 1
    temp = round_num

    # 提取 feature 部分
    feature_part = parts[2].split(':')[1].strip()
    feature_list = list(map(float, feature_part.strip().strip('[]').split(',')))
    # 提取 act 部分
    act_part = parts[1].split(':')[1].strip()

    act_list = [list(map(int, pair.strip('[]').split(','))) for pair in act_part.split('], [')]

    return [feature_list, round_num, act_list]


def read_optimal_solutions(file_path):
    optimal_solutions = []

    with open(file_path, 'r') as file:
        for line in file:
            optimal_solutions.append(parse_line(line))

    return optimal_solutions


def mergePareto_set():
    global times
    file_path = 'pareto_set.txt'

    optimal_solutions = read_optimal_solutions(file_path)
    avg_paretonum = len(optimal_solutions) / times
    templen = len(optimal_solutions)
    # 合并得到pareto
    optimal_solutions, merged_count, same_count = mergePareto(optimal_solutions)
    with open("pareto_final.txt", "a") as f:
        for a in optimal_solutions:
            f.write("round:" + str(a[1]) + "; act:" + str(a[2]) + "; feature:" + str(a[0]) + "\n")

    text=(" 训练集合数" + str(times) +
          " 原先平均条数：" + str(templen / times) +
          " 现在条数：" + str(len(optimal_solutions)) +
          " 最优解重复次数"+ str(same_count) +
          " 最优解重复次率" + str(same_count/templen) +
          " 合并保留率(现在条数/原先平均条数，大于1)：" + str(len(optimal_solutions)* times / templen)+
          " \n" )
    print(text)
    with open("pareto_merge.txt", "a") as f:
        f.write(text)
    return optimal_solutions


def mergePareto_final():
    global times
    file_path = 'pareto_final.txt'

    optimal_solutions = read_optimal_solutions(file_path)

    # 合并得到pareto
    optimal_solutions,_, _ = mergePareto(optimal_solutions)
    os.remove(file_path)
    with open("pareto_final.txt", "a") as f:
        f.write("CPUnum="+str(CPUnum)+"Mem="+str(Mem)+"BandWith="+str(BandWidth)+"policy="+policy_+"\n")
        for a in optimal_solutions:
            f.write("round:" + str(a[1]) + "; act:" + str(a[2]) + "; feature:" + str(a[0]) + "\n")

    with open("pareto_merge.txt", "a") as f:
        f.write("pareto大小："+str(len(optimal_solutions)))
    return optimal_solutions


def calActionTimesFromPareto():
    file_path = 'pareto_final.txt'
    optimal_solutions = read_optimal_solutions(file_path)
    for o in optimal_solutions:
        o[2]=sorted(o[2], key=lambda x: x[1])
        for o in optimal_solutions:
            o[2] = sorted(o[2], key=lambda x: x[1])
    action_list=[]
    index=0
    for solution in optimal_solutions:
        existed=False
        if index!=0:
            for action in action_list:
                if action[0][2]==solution[2]:
                    existed=True
                    action.append(solution)
                    break
        if not existed:
            index+=1
            action_list.append([solution])

    print(action_list)
    print(index)

    return
if __name__ == "__main__":
    pareto_set = mergePareto_set()
    mergePareto_final()
    #calActionTimesFromPareto()
    print("pareto_final解析完成")
    showPareto(pareto_set)

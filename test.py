import random

glo=1
def test1():
    List=[[2,3]]

    if List[0]:
        print("1")
    List.append(0)
    List.append(1)
    if List[0]==[]:
        print("2")
    if List[1]:
        print("3")
    if List[2]:
        print("4")


    print(List)
def rpm():
    print(rpm)
def test2():
    step=1
    flag1=0
    mini=-1
    co=0
    allCost = [[], [], [], [], [], []]
    while True:
        cost=random.randint(-10,100)

        if allCost[step - 1]:
            # allCost = [[], [], [], [], [], []]
            # 如果内层list中有值，则mini取其中最小值，否则为-1
            # 每个列表代表一个step
            mini = min(allCost[step - 1])
        if flag1 == 0:
            # if it's the first episode, save the cost directly
            if cost > 0:
                allCost[step - 1].append(cost)
                reward = 0
                co += 1
            else:
                flag1 += 1
                for i in range(co):
                    allCost[step - 1 - (i + 1)].clear()
                break
        else:
            if cost > 0:
                if step == 6:
                    if abs(min(allCost[step - 1]) - cost) < 0.0001:
                        reward = test_reward
                    elif (min(allCost[step - 1]) - cost) > 0:
                        test_reward = test_reward + 100
                        reward = test_reward
                    else:
                        reward = 10 * (min(allCost[step - 1]) - cost)
                    for i in range(6):
                        rpm()
                    allCost[step - 1].append(cost)
            else:
                reward = -100
                rpm()


def test3():
    import numpy as np

    pred_q = np.array([3.2, 1.5,4.1,2.7 ])
    sorted_indices = np.argsort(pred_q)
    print(sorted_indices)
    # sorted_indices 现在是 [1, 2, 0, 3]

def test4():
    return 0.1,0.2,0.3

def test5():
    a=1
    b=2
    c=-1
    if a>0 and b>0:
        print(1)
    elif a>0 and c>0:
        print(2)
def test6():
    global glo
    glo += 1
    print(glo)

if __name__ == '__main__':

    test6()
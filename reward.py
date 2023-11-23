times = 100
from weight import getWeightReward
#from train import ep
ep=0
min1=0
min2=0
min3=0
max1=0
max2=0
max3=0
def reward1cal(feature,min,max):

    reward=0
    if ep==1:
        if feature<0:
            return -100,75,75
        else:
            return 0,feature,feature
    if feature<0:
        return -100,min,max
    else:
        if min==max==feature:
            return 0,feature,feature
        if feature<min:
            min=feature
            reward+=1
        elif feature>max:
            max=feature
            reward-=1

        reward=reward+(max-feature)/(max-min)


    return reward,min,max


def reward2cal(feature,min,max):

    reward = 0
    if ep==1:
        return 0,feature,feature
    else:
        if min==max==feature:
            return 0,feature,feature
        if feature<min:
            min=feature
            reward+=1
        elif feature>max:
            max=feature
            reward-=1

        reward=reward+(max-feature)/(max-min)
    return reward,min,max


def reward3cal(feature,min,max):
    global min3
    global max3
    reward = 0
    if ep == 1:
        return 0, feature, feature
    else:
        if min==max==feature:
            return 0,feature,feature
        if feature<min:
            min=feature
            reward+=1
        elif feature>max:
            max=feature
            reward-=1

        reward=reward+(max-feature)/(max-min)
    return reward,min,max



def Culreward(f1, f2, f3):
    global min1,min2,min3,max1,max2,max3, ep

    ep+=1
    r1,min1,max1 = reward1cal(f1,min1,max1)
    r2,min2,max2 = reward2cal(f2,min2,max2)
    r3,min3,max3 = reward3cal(f3,min3,max3)



    return r1, r2, r3

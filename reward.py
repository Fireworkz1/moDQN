times = 10
from weight import getWeightReward

avg_feature1 = 0
avg_feature2 = 0
avg_feature3 = 0

ep = 0
ep1 = 0


def reward1cal(feature, avg_f, ep):
    global ep1

    if feature < 0:
        reward = -10000
        return reward, avg_f
    ep1 += 1
    if ep1 == 1:
        avg_f = feature
    if feature < avg_f:
        reward = 50 + 100 * (avg_f - feature)
    else:
        reward = 100 * (avg_f - feature)
    avg_f = (avg_f * (ep1 - 1) + feature) / ep1
    return reward, avg_f


# if feature < 0:
#     reward = -200
# elif abs(feature - min) < 0.0001:
#     reward = 0
# elif feature < min:
#     reward = 100
#     min = feature
# else:
#     reward = times * (min - feature)
# return reward, min


def reward2cal(feature, avg_f, ep):
    if ep == 1:
        avg_f = feature
    if feature < avg_f:
        reward = 50 + 100 * (avg_f - feature)
    else:
        reward = 100 * (avg_f - feature)
    avg_f = (avg_f * (ep - 1) + feature) / ep
    return reward, avg_f


# if feature < 0:
#     reward = -100
# elif abs(feature - min) < 0.0001:
#     reward = 0
# elif feature < min:
#     reward = 100
#     min = feature
# else:
#     reward = times* (min - feature)
# return reward, min


def reward3cal(feature, avg_f, ep):
    if ep == 1:
        avg_f = feature

    if feature < avg_f:
        reward = 50 + 100 * (avg_f - feature)
    else:
        reward = 100 * (avg_f - feature)
    avg_f = (avg_f * (ep - 1) + feature) / ep
    return reward, avg_f


# if feature < 0:
#     reward = -100
# elif abs(feature - min) < 0.0001:
#     reward = 0
# elif feature < min:
#     reward = 100
#     min = feature
# else:
#     reward = times * (min - feature)
# return reward, min


def Culreward(f1, f2, f3, mf1, mf2, mf3):
    global ep
    global avg_feature1
    global avg_feature2
    global avg_feature3

    ep += 1

    r1, avg_feature1 = reward1cal(f1, avg_feature1, ep)
    r2, avg_feature2 = reward2cal(f2, avg_feature2, ep)
    r3, avg_feature3 = reward3cal(f3, avg_feature3, ep)

    w = getWeightReward()
    r1 = r1 * w[0]
    r2 = r2 * w[1]
    r3 = r3 * w[2]

    return r1, r2, r3, avg_feature1, avg_feature2, avg_feature3

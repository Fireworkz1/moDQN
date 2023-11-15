def reward1cal(feature, min):
    if feature < 0:
        reward = -100
    elif abs(feature - min) < 0.0001:
        reward = 0
    elif feature < min:
        reward = 100
        min = feature
    else:
        reward = 10 * (min - feature)
    return reward, min


def reward2cal(feature, min):
    if feature < 0:
        reward = -100
    elif abs(feature - min) < 0.0001:
        reward = 0
    elif feature < min:
        reward = 100
        min = feature
    else:
        reward = 10 * (min - feature)
    return reward, min


def reward3cal(feature, min):
    if feature < 0:
        reward = -100
    elif abs(feature - min) < 0.0001:
        reward = 0
    elif feature < min:
        reward = 100
        min = feature
    else:
        reward = 10 * (min - feature)
    return reward, min


def Culreward(f1, f2, f3, mf1, mf2, mf3):
    r1, mf1 = reward1cal(f1, mf1)
    r2, mf2 = reward2cal(f2, mf2)
    r3, mf3 = reward3cal(f3, mf3)
    return r1, r2, r3, mf1, mf2, mf3

def dominates(solution1, solution2):
    # 辅助函数，判断 solution1 是否支配 solution2
    return all(s1 >= s2 for s1, s2 in zip(solution1, solution2)) and any(
        s1 > s2 for s1, s2 in zip(solution1, solution2))


def calPareto(solution_set, new_solution):
    new_solution_set = []
    removed_solutions = []
    if new_solution[0][0] < 0:
        return False, solution_set, []
    for existing_solution in solution_set:
        if dominates(existing_solution[0], new_solution[0]):
            removed_solutions.append(existing_solution)
        elif dominates(new_solution[0], existing_solution[0]):
            continue
        else:
            new_solution_set.append(existing_solution)

    if all(not dominates(new_solution[0], sol[0]) for sol in solution_set) and new_solution[0] not in [row[0] for row in
                                                                                                       new_solution_set]:
        new_solution_set.append(new_solution)
        return True, new_solution_set, removed_solutions
    else:
        return False, solution_set, []


def getPareto(feature_set, pareto_set, action_list):
    # set元素格式：[f1,f2,f3],round,[[],[],[],[],[],[]]]
    new_solution = feature_set[-1]
    ispareto, pareto_set, removed_set = calPareto(pareto_set, new_solution)

    return ispareto, pareto_set, removed_set


def mergePareto(solution_set):
    pareto_set = []
    merged_count = 0
    same_count = 0
    for new_solution in solution_set:
        is_dominated = False

        # 检查新解是否被 Pareto 前沿中的解所支配
        for existing_solution in pareto_set:
            if dominates(existing_solution[0], new_solution[0]):
                is_dominated = True
                merged_count += 1
                break

        if not is_dominated:
            # 如果新解不被支配，则将其添加到 Pareto 前沿
            pareto_set = [sol for sol in pareto_set if not dominates(new_solution[0], sol[0])]
            if new_solution[0] not in [row[0] for row in pareto_set]:
                pareto_set.append(new_solution)
            else:
                same_count += 1

    return pareto_set, merged_count,same_count



def cardinality(relations, selectivities):
    if len(relations) == 1:
        return relations[0]["cardinality"]
    result = 1
    i = 0
    j = 1
    while len(relations) > i + 1:
        result *= selectivities[i][j] * relations[i]["cardinality"]
        i += 1
        j += 1
    return result


def standard_cost_left_deep(relations, selectivities):
    if len(relations) == 1:
        return cardinality(relations, selectivities)
    current_cardinality = cardinality(relations, selectivities)
    left_branch = [relations[0]]
    right_branch = relations[1:]
    return current_cardinality + standard_cost_left_deep(left_branch, selectivities) + standard_cost_left_deep(right_branch, selectivities)


def weight_function(relation, relations = None, selectivities = None):
    if relations == None and selectivities == None:
        return relation["cardinality"]
    else:
        return standard_cost_left_deep([relation] + relations, selectivities)
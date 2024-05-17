def flatten(lst):
    for i in lst:
        if isinstance(i, list):
            yield from flatten(i)
        else:
            yield i

def join_tree_cardinality(join_tree, relations, selectivities):
    if type(join_tree) == int:
        return relations[join_tree]["cardinality"]
    
    selectivity = 1
    if type(join_tree[0]) == list:
        flattened_relations_left = list(flatten(join_tree[0]))
    else:
        flattened_relations_left = [join_tree[0]]
        
    if type(join_tree[1]) == list:
        flattened_relations_right = list(flatten(join_tree[1]))
    else:
        flattened_relations_right = [join_tree[1]]    
    
    for rel_l in flattened_relations_left:
        for rel_r in flattened_relations_right:
            x, y = sorted([rel_l, rel_r])
            if (x, y) in selectivities:
                selectivity *= selectivities[(x, y)]["selectivity"]
    
    result = selectivity * join_tree_cardinality(join_tree[0], relations, selectivities)*join_tree_cardinality(join_tree[1], relations, selectivities)
    return result


def basic_cost(join_tree, relations, selectivities):
    if type(join_tree) == int:
        return 0
    current_cardinality = join_tree_cardinality(join_tree, relations, selectivities)
    return current_cardinality + basic_cost(join_tree[0], relations, selectivities) + basic_cost(join_tree[1], relations, selectivities)
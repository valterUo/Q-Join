import itertools
from classical_algorithms.weights_costs import standard_cost_left_deep


def create_join_tree(T1, T2, selectivities, join_methods, left_deep_only = False, right_deep_only = False):
    costs = {}
    if left_deep_only:
        for method in join_methods:
            costs[method["name"]] = method["impl"]([T1, T2], selectivities)
    elif right_deep_only:
        for method in join_methods:
            costs[method["name"]] = method["impl"]([T2, T1], selectivities)
    else:
        for method in join_methods:
            costs[method["name"] + "_left"] = method["impl"]([T1, T2], selectivities)
            costs[method["name"] + "_right"] = method["impl"]([T2, T1], selectivities)
            
    # Return method with minimum cost
    return min(costs, key = costs.get)


def dynamic_programming(relations, selectivities, join_methods, allow_cross_products = False, left_deep_only = False, right_deep_only = False):
    n = len(relations)
    dp_table = {}
    for rel in relations:
        dp_table[frozenset([rel])] = rel
    
    # for each 1 < s ≤ n ascending
    for s in range(n, 1, -1):
        
        # For each subset of size s - 1
        for subset in itertools.combinations(relations, s - 1):
            
            subset = frozenset(subset)
            
            if subset not in dp_table:
                continue
            
            # For each relation not in the subset
            for rel in relations:
                if rel not in subset:
                    p1 = dp_table[subset]
                    p2 = dp_table[frozenset([rel])]
                    CP = create_join_tree(p1, p2, selectivities, join_methods, left_deep_only, right_deep_only)
                    join = frozenset([rel]).union(subset)
                    if join not in dp_table[join] or CP < standard_cost_left_deep(join):
                        dp_table[join] = CP
    
    return dp_table[frozenset(relations)]
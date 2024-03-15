def GreedyJoinOrdering1(relations, w):
    ordered_joins = []
    while len(relations) > 1:
        min_weight = float('inf')
        min_relation = None
        for relation in relations:
            weight = w(relation)
            if weight < min_weight:
                min_weight = weight
                min_relation = relation
        ordered_joins.append(min_relation)
        relations.remove(min_relation)
    return ordered_joins


def GreedyJoinOrdering2(relations, w):
    ordered_joins = []
    while len(relations) > 1:
        min_weight = float('inf')
        min_relation = None
        for relation in relations:
            weight = w(relation, ordered_joins)
            if weight < min_weight:
                min_weight = weight
                min_relation = relation
        ordered_joins.append(min_relation)
        relations.remove(min_relation)
    return ordered_joins


def GreedyJoinOrdering3(relations, w):
    ordered_joins = []
    for relation in relations:
        relations_prime = relations.copy()
        relations_prime.remove(relation)
        ordered_joins_prime = [relation] + GreedyJoinOrdering2(relations_prime, w)
        ordered_joins.append(ordered_joins_prime)
        
    join_min_weight = float('inf')
    join_min = None
    for join_options in ordered_joins:
        join_options_weight = sum([w(join_options[i], join_options[1:i]) for i in range(1, len(join_options))])
        if join_options_weight < join_min_weight:
            join_min_weight = join_options_weight
            join_min = join_options
    
    return join_min
import itertools

import networkx as nx

from classical_algorithms.weights_costs import basic_cost


def create_join_tree(T1, T2, relations, selectivities):
    cost1 = basic_cost([T1, T2], relations, selectivities)
    cost2 = basic_cost([T2, T1], relations, selectivities)
    if cost1 < cost2:
        return [T1, T2], cost1
    return [T2, T1], cost2


def dynamic_programming(relations, selectivities):
    n = len(relations)
    dp_table = {}
    
    for rel in relations:
        dp_table[frozenset([rel])] = rel
    
    # for each 1 < s ≤ n ascending
    for s in range(2, n + 1):
        
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
                    T, T_cost = create_join_tree(p1, p2, relations, selectivities)
                    join_key = frozenset(subset.union({rel}))
                    
                    if join_key not in dp_table:
                        dp_table[join_key] = T
                    elif basic_cost(dp_table[join_key], relations, selectivities) > T_cost:
                        dp_table[join_key] = T
    
    return dp_table[frozenset(relations)]


def all_connected_subgraphs(g, m):
    def _recurse(t, possible, excluded):
        if len(t) == m:
            yield t
        else:
            excluded = set(excluded)
            for i in possible:
                if i not in excluded:
                    new_t = (*t, i)
                    new_possible = possible | g[i]
                    excluded.add(i)
                    yield from _recurse(new_t, new_possible, excluded)
    excluded = set()
    for (i, possible) in enumerate(g):
        excluded.add(i)
        yield from _recurse((i,), possible, excluded)


def graph_aware_dynamic_programming(query_graph, relations, selectivities):
    n = len(relations)
    dp_table = {}
    
    adj_list_dict = nx.to_dict_of_lists(query_graph)
    adj_list = [set(adj_list_dict[node]) for node in sorted(adj_list_dict.keys())]
    
    for rel in relations:
        dp_table[frozenset([rel])] = rel
    
    # for each 1 < s ≤ n ascending
    for s in range(2, n + 1):
        
        # For each subset of size s - 1
        for subset in all_connected_subgraphs(adj_list, s - 1):
            subset = frozenset(subset)
            
            if subset not in dp_table:
                continue
            
            # For each relation not in the subset
            for rel in relations:
                if rel not in subset:
                    # Check that rel is connected in the query graph to some relation in the subset
                    #connected = False
                    #for r in subset:
                    #    if query_graph.has_edge(rel, r):
                    #        connected = True
                    #        break
                    #if connected:
                    p1 = dp_table[subset]
                    p2 = dp_table[frozenset([rel])]
                    T, T_cost = create_join_tree(p1, p2, relations, selectivities)
                    join_key = frozenset(subset.union({rel}))
                    
                    if join_key not in dp_table:
                        dp_table[join_key] = T
                    elif basic_cost(dp_table[join_key], relations, selectivities) > T_cost:
                        dp_table[join_key] = T
    
    return dp_table[frozenset(relations)]
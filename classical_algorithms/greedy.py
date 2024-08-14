from itertools import combinations

from classical_algorithms.weights_costs import basic_cost


def greedy(query_graph, relations, selectivities):
    join_result = []
    tables = query_graph.nodes()
    min_cost = float('inf')
    
    #relations = {table: {"cardinality": table["cardinality"]} for table in tables}
    #selectivities = {(table1, table2): {"selectivity": query_graph[table1][table2]["selectivity"]} for table1, table2 in query_graph.edges()}
    
    for table1, table2 in query_graph.edges():
        selectivity = query_graph[table1][table2]["selectivity"]
        current_cost = relations[table1]["cardinality"] * relations[table2]["cardinality"]*selectivity
        if current_cost < min_cost:
            min_cost = current_cost
            join_result = [table1, table2]
    
    joined_tables = [table for table in join_result]
    
    for _ in range(len(query_graph.edges) - 1):
        min_cost = float('inf')
        current_min_table = None
        for table in tables:
            if table not in joined_tables:
                current_join_tree = [table, join_result]
                current_cost = basic_cost(current_join_tree, relations, selectivities)
                if current_cost < min_cost:
                    min_cost = current_cost
                    current_min_table = table
        if current_min_table is not None:
            joined_tables.append(current_min_table)
            join_result = [current_min_table, join_result]
    
    return join_result


def greedy_with_query_graph(query_graph, relations, selectivities):
    join_result = []
    min_cost = float('inf')
    
    for table1, table2 in query_graph.edges():
        selectivity = query_graph[table1][table2]["selectivity"]
        current_cost = relations[table1]["cardinality"] * relations[table2]["cardinality"]*selectivity
        if current_cost < min_cost:
            min_cost = current_cost
            join_result = [table1, table2]
    
    joined_tables = [table for table in join_result]
    
    for _ in range(len(query_graph.edges) - 1):
        
        adjacent_tables = []
        for table in joined_tables:
            adjacent_tables += [table2 for table2 in query_graph[table] if table2 not in joined_tables]
        
        min_cost = float('inf')
        current_min_table = None
        
        for table in adjacent_tables:
            current_join_tree = [table, join_result]
            current_cost = basic_cost(current_join_tree, relations, selectivities)
            if current_cost < min_cost:
                min_cost = current_cost
                current_min_table = table
                
        if current_min_table is not None:
            joined_tables.append(current_min_table)
            join_result = [current_min_table, join_result]
    
    return join_result


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
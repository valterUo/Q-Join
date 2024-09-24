from itertools import combinations
import json
import math
import os
import numpy as np

from classical_algorithms.weights_costs import basic_cost, join_tree_cardinality


class Variable:
    
    def __init__(self, relations, selectivities, new_subgraph, rel1, rel2, level, base_labeling):
        
        self.level = level
        self.local_cost = 1
        self.new_subgraph = new_subgraph
        
        tables = list(new_subgraph)
        join_tree = [tables[0], tables[1]]
        for rel in tables[2:]:
            join_tree = [rel, join_tree]

        self.local_cost = join_tree_cardinality(join_tree, relations, selectivities)
        
        self.labeling = base_labeling + [(rel1, rel2, level)]
        self.rel1 = rel1
        self.rel2 = rel2
        
    def get_labeling(self):
        return self.labeling
    
    def get_local_cost(self):
        return self.local_cost
    
    def get_level(self):
        return self.level
    
    def __str__(self) -> str:
        return f'Variable({self.rel1}, {self.rel2}, {self.level}) with labelings {self.labeling} and local cost {self.local_cost}'



# Encodes, for example, the following constraint:
# (x - a - b - c)^2 = a^2 + 2 a b + 2 a c - 2 a x + b^2 + 2 b c - 2 b x + c^2 - 2 c x + x^2
def combinations_with_variable(x, vars, scaler = 1):
    result = {}
    result[(x,)] = scaler
    for v in vars:
        result[(v,)] = scaler
        result[(x, v)] = -2 * scaler
    
    for comb in combinations(vars, 2):
        result[comb] = 2 * scaler
    
    return result

# Encodes, for example, the following constraint:
# (1 + 2*x_2 + ... + max_number_of_levels * x_n - sum over table variables)^2
# Uses also trick from Lucas how to encode integer variables
def table_number_constraint(max_number_of_levels, tables, table_id, scaler = 1):
    integer_vars = []
    N = max_number_of_levels
    M = int(np.floor(np.log2(N)))
    result = {}
    
    for i in range(1, M + 1):
        if i == M:
            integer_vars.append((f'table_{table_id}_{i}', (N + 1 - 2**M)))
        integer_vars.append((f'table_{table_id}_{i}', 2**i))
    
    for int_var in integer_vars:
        result[(int_var[0],)] = scaler * (int_var[1]**2 + 2 * int_var[1])
    
    for table in tables:
        result[(table,)] = -scaler
    
    for x, y in combinations(integer_vars + tables, 2):
        if x in integer_vars and y in integer_vars:
            result[(x[0], y[0])] = 2 * scaler * x[1] * y[1]
        elif x in integer_vars and y in tables:
            result[(x[0], y)] = -2 * scaler * x[1]
        elif x in tables and y in integer_vars:
            result[(x, y[0])] = -2 * scaler * y[1]
        else:
            result[(x, y)] = 2 * scaler
    
    return result
    
    
def get_connected_subgraphs_with_dfs(graph, node, n, approximate=False, excluded_node = None):
    connected_subgraphs = set()
    
    # Function to perform DFS to find connected subgraphs of size n
    def dfs(start_node, path):
        
        if len(path) == n + 1:
            if excluded_node in path:
                return
            connected_subgraphs.add(tuple(path))
            return
        
        for r in range(1, n - len(path) + 2):
            
            combs = combinations(graph.neighbors(start_node), r)
            
            if approximate:
                neighbors_combination = next(combs)
                neighbor = neighbors_combination[0]
                if neighbor not in path:
                    dfs(neighbor, path.union(set(neighbors_combination)))
            else:
                for neighbors_combination in combs:
                    for neighbor in neighbors_combination:
                        if neighbor not in path:
                            dfs(neighbor, path.union(set(neighbors_combination)))
            
    for r in range(1, n + 1):
        
        combs = combinations(graph.neighbors(node), r)
        
        if approximate:
            neighbors_combination = next(combs)
            neighbor = neighbors_combination[0]
            dfs(neighbor, set(list(neighbors_combination) + [node]))
        else:
            for neighbors_combination in combs:
                for neighbor in neighbors_combination:
                    dfs(neighbor, set(list(neighbors_combination) + [node]))
    
    return connected_subgraphs


def build_nested_list(tuples):
    children_map = {}

    for parent, current, level in tuples:
        if level not in children_map:
            children_map[level] = []
        children_map[level].append((parent, current))

    levels = list(sorted(children_map.keys()))
    nested_list = []
    added_nodes = set()
    for i in levels:
        for parent, current in children_map[i]:
            if parent not in added_nodes and current not in added_nodes:
                if nested_list == []:
                    nested_list = sorted([parent, current])
                else:
                    nested_list= [sorted([parent, current]), nested_list]
                added_nodes.add(parent)
                added_nodes.add(current)
            elif parent not in added_nodes:
                nested_list = [parent, nested_list]
                added_nodes.add(parent)
            elif current not in added_nodes:
                nested_list = [current, nested_list]
                added_nodes.add(current)
                
    return nested_list


def store_gurobi_results(results, time, file, qjoin):
    
    gurobi_res = {}
    for var in results["result"]:
        if results["result"][var] == 1 and "a" not in var:
            var = "".join(var.split("_"))
            gurobi_res[eval(var)] = 1
    
    gurobi_res_pos = gurobi_res.copy()
    gurobi_res_pos = {str(k): v for k, v in gurobi_res_pos.items() if v == 1}
    
    tuples = list(gurobi_res.keys())
    nest = build_nested_list(tuples)
    print(nest)
    for v in qjoin.full_hubo.variables:
        if v not in gurobi_res:
            gurobi_res[v] = 0
            
    quantum_cost = qjoin.evaluate_cost(gurobi_res)
    stored_result = {"result": gurobi_res_pos, 
                           "solution": nest, 
                           "cost": quantum_cost, 
                           "cost_with_classical": basic_cost(nest, qjoin.relations, qjoin.selectivities), 
                           "time": time}
    
    with open(file, 'w') as f:
        json.dump(results, f)
        
        
def append_to_json(file, key, data):
    
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump({}, f)
    
    with open(file, 'r') as f:
        json_data = json.load(f)
    
    #json_data[key] = data
    if len(json_data) == 1:
        key = list(json_data.keys())[0]
    
    if key not in json_data:
        json_data[key] = data
    else:
        old_data = json_data[key]
        # append data to the old data
        for k in old_data:
            if k in data:
                if isinstance(old_data[k], list):
                    pass
                    #old_data[key].extend(data[key])
                else:
                    old_data[k] = old_data[k] + data[k]
            else:
                old_data[k] = data[k]
        json_data[key] = old_data
    
    with open(file, 'w') as f:
        json.dump(json_data, f, indent=4)
        
        
def compare_nested_lists(list1, list2):
    # If the input is not lists, compare directly
    if not isinstance(list1, list) and not isinstance(list2, list):
        return list1 == list2
    
    # If the input is only one list, they can't be equal
    if not isinstance(list1, list) or not isinstance(list2, list):
        return False
    
    # Sort the elements within each sublist
    sorted_list1 = sorted(list1, key=lambda x: str(x))
    sorted_list2 = sorted(list2, key=lambda x: str(x))
    
    # Sort the sublists
    sorted_list1.sort(key=lambda x: str(x))
    sorted_list2.sort(key=lambda x: str(x))
    
    # Now compare the sorted lists recursively
    if len(sorted_list1) != len(sorted_list2):
        return False
    
    for sublist1, sublist2 in zip(sorted_list1, sorted_list2):
        if not compare_nested_lists(sublist1, sublist2):
            return False
    
    return True

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
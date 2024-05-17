from classical_algorithms.weights_costs import join_tree_cardinality


class Variable:
    
    def __init__(self, relations, selectivities, new_subgraph, subgraph, rel1, rel2, level, base_labeling):
        
        self.subgraph = subgraph
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
    
    def __str__(self) -> str:
        return f'Variable({self.rel1}, {self.rel2}, {self.level}) with labelings {self.labeling} and local cost {self.local_cost}'
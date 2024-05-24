import networkx as nx
from networkx.generators.trees import random_tree
import numpy as np
np.random.seed(0)

# Note: 1 - np.random.uniform(0, 1) draws a random number from (0, 1] (wanted)
# whereas np.random.uniform(0, 1) would draw a random number from [0, 1) (wrong)

class QueryGraphs:
    
    def __init__(self):
        pass
    
    # Interval 4 - 9 for precise
    # Interval 4 - ? for approximate
    def tree(self, n_nodes = 10):
        query_graph = random_tree(n_nodes, seed=0)
        np.random.seed(0)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['label'] = 'R' + str(i)
            query_graph.nodes[i]['cardinality'] = np.random.randint(10, 50)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = 1 - np.random.uniform(0, 1)
            
        return query_graph
    
    # Interval 3 - 12 for precise
    # Interval 3 - ? for approximate
    def chain(self, n_nodes=10):
        np.random.seed(0)
        query_graph = nx.path_graph(n_nodes)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['cardinality'] = np.random.randint(10, 50)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = 1 - np.random.uniform(0, 1)
            
        return query_graph
    
    # Interval 3 - 5 for precise
    # Interval 3 - ? for approximate
    def clique(self, n_nodes=10):
        np.random.seed(0)
        query_graph = nx.complete_graph(n_nodes)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['cardinality'] = np.random.randint(10, 50)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = 1 - np.random.uniform(0, 1)
            
        return query_graph
    
    
    def cycles(self, n_nodes=10):
        np.random.seed(0)
        query_graph = nx.cycle_graph(n_nodes)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['cardinality'] = np.random.randint(10, 50)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = 1 - np.random.uniform(0, 1)
            
        return query_graph
    
    
    def star(self, n_nodes=10):
        np.random.seed(0)
        query_graph = nx.star_graph(n_nodes - 1)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['cardinality'] = np.random.randint(10, 50)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = 1 - np.random.uniform(0, 1)
            
        return query_graph
    
    
    def random(self, n_nodes=10):
        np.random.seed(0)
        query_graph = nx.gnp_random_graph(n_nodes, 0.55, directed=False)
        
        while True:
            if nx.is_connected(query_graph):
                break
            query_graph = nx.gnp_random_graph(n_nodes, 0.55, directed=False)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['cardinality'] = np.random.randint(1, 30)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = 1 - np.random.uniform(0, 1)
            
        return query_graph
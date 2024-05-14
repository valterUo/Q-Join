import networkx as nx
from networkx.generators.trees import random_tree
import numpy as np
np.random.seed(0)

class QueryGraphs:
    
    def __init__(self):
        pass
        
    def get_graph_1(self):
        query_graph = nx.Graph()
        
        query_graph.add_node(0, label ='R1', cardinality = 10)
        query_graph.add_node(1, label='R2', cardinality = 100)
        query_graph.add_node(2, label='R3', cardinality = 1000)
        
        query_graph.add_edge(0, 1, label='0', selectivity = 0.1)
        query_graph.add_edge(1, 2, label='1', selectivity = 0.2)
        
        return query_graph
        
    def get_graph_2(self):
        query_graph = nx.Graph()
                
        query_graph.add_node(0, label ='R1', cardinality = 10)
        query_graph.add_node(1, label='R2', cardinality = 100)
        query_graph.add_node(2, label='R3', cardinality = 1000)
        query_graph.add_node(3, label='R4', cardinality = 10000)
        query_graph.add_node(4, label='R5', cardinality = 100000)
        query_graph.add_node(5, label='R6', cardinality = 1000000)
        query_graph.add_node(6, label='R7', cardinality = 10000000)
        
        query_graph.add_edge(0, 1, label='0', selectivity = 0.1)
        query_graph.add_edge(1, 2, label='1', selectivity = 0.2)
        query_graph.add_edge(2, 3, label='2', selectivity = 0.3)
        query_graph.add_edge(3, 4, label='3', selectivity = 0.4)
        query_graph.add_edge(4, 5, label='4', selectivity = 0.5)
        query_graph.add_edge(5, 6, label='5', selectivity = 0.6)
        
        return query_graph
    
    def get_graph_3(self):
        query_graph = nx.Graph()
                
        query_graph.add_node(0, label ='R1', cardinality = 100)
        query_graph.add_node(1, label='R2', cardinality = 10)
        query_graph.add_node(2, label='R3', cardinality = 10)
        query_graph.add_node(3, label='R4', cardinality = 10)
        
        query_graph.add_edge(0, 1, label='0', selectivity = 0.1)
        query_graph.add_edge(1, 2, label='1', selectivity = 0.0001)
        query_graph.add_edge(2, 3, label='2', selectivity = 0.1)
        
        return query_graph
    
    def get_graph_4(self):
        nodes = 10
        query_graph = nx.complete_graph(nodes)
        
        for i in range(nodes):
            query_graph.nodes[i]['label'] = 'R' + str(i)
            query_graph.nodes[i]['cardinality'] = 2 * (i+1)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = 0.001*(l+1)
            
        return query_graph
    
    def get_graph_5(self):
        query_graph = nx.Graph()
        
        query_graph.add_node(0, label ='R1', cardinality = 1000)
        query_graph.add_node(1, label='R2', cardinality = 2)
        query_graph.add_node(2, label='R3', cardinality = 2)
        
        query_graph.add_edge(0, 1, label='0', selectivity = 0.1)
        query_graph.add_edge(0, 2, label='1', selectivity = 0.1)
        
        return query_graph
    
    def get_graph_6(self):
        query_graph = nx.Graph()
        
        query_graph.add_node(0, label ='R1', cardinality = 10)
        query_graph.add_node(1, label='R2', cardinality = 20)
        query_graph.add_node(2, label='R3', cardinality = 20)
        query_graph.add_node(3, label='R3', cardinality = 10)
        
        query_graph.add_edge(0, 1, label='0', selectivity = 0.01)
        query_graph.add_edge(1, 2, label='1', selectivity = 0.5)
        query_graph.add_edge(2, 3, label='2', selectivity = 0.01)
        
        return query_graph
    
    def get_graph_7(self, n_nodes=10):
        query_graph = nx.complete_graph(n_nodes)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['label'] = 'R' + str(i)
            query_graph.nodes[i]['cardinality'] = np.random.randint(10, 1000)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = np.random.uniform(0, 1)
            
        return query_graph
    
    def get_graph_8(self, n_nodes = 10):
        query_graph = random_tree(n_nodes, seed=0)
        
        for i in range(n_nodes):
            query_graph.nodes[i]['label'] = 'R' + str(i)
            query_graph.nodes[i]['cardinality'] = np.random.randint(10, 1000)
        
        for l, (i, j) in enumerate(query_graph.edges):
            query_graph.edges[i, j]['label'] = l
            query_graph.edges[i, j]['selectivity'] = np.random.uniform(0, 1)
            
        return query_graph
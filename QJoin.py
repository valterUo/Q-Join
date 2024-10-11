from itertools import combinations
import math
import time
import cplex
import dimod
import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from dwave.system import LeapHybridSampler
from dwave.system import EmbeddingComposite, DWaveSampler
from hybrid.reference import KerberosSampler
from dwave.samplers import TabuSampler, SimulatedAnnealingSampler, SteepestDescentSampler
from math import comb as ncr

from classical_algorithms.dynamic_programming import dynamic_programming, graph_aware_dynamic_programming
from classical_algorithms.greedy import greedy, greedy_with_query_graph
from classical_algorithms.weights_costs import basic_cost
from qaoa.qaoa import QuantumApproximateOptimizationAlgorithm
from utils import combinations_with_variable, is_chain_tree, is_star_graph, table_number_constraint, Variable

class QJoin:
    
    def __init__(self, query_graph, query_graph_name, scaler=1, hubo_to_bqm_strength=5, method_name = "precise_1", create_bqm = True, estimation_size = 2):
        self.query_graph = query_graph
        self.query_graph_name = query_graph_name
        self.scaler = scaler
        self.max_number_of_ranks = len(query_graph.nodes) - 1
        self.estimation_size = estimation_size
        self.ranks = range(self.max_number_of_ranks)

        self.relations = {}
        for node in query_graph.nodes(data=True):
            self.relations[node[0]] = node[1]

        self.selectivities = {}
        for edge in query_graph.edges(data=True):
            self.selectivities[(edge[0], edge[1])] = edge[2]
        
        self.variables = {}
        self.variables_dict = {}
        self.normalized_variable_dict = {}
        self.hubo = None
        self.hubo_total_cost = None
        self.hubo_variables = None
        self.variables_by_rank = {}
        self.variables_by_joins = {}
        self.validity_constraints = {}
        self.samplesets = {}
        self.lp_file = "lp_files//" + str(query_graph) + "_qubo.lp"
        self.name = str(query_graph)
        self.offset = 0
        
        
        if method_name == "precise_1":
            self.construct_presice_cost_function()
            self.group_variables()
            self.every_rank_appears_exactly_once()
            self.hubo_combinations()
            
        elif method_name == "heuristic_1":
            self.construct_estimate_cost_function()
            self.group_variables()
            
            self.hubo_total_cost.normalize()
            sample_like = {}
            for var in self.hubo_variables:
                sample_like[var] = 1
            max_cost = self.hubo_total_cost.energy(sample_like)   
                
            self.scaler = max_cost
            
            self.every_rank_appears_exactly_once()
            self.hubo_combinations()
            
        elif method_name == "precise_2":
            self.construct_presice_cost_function()
            self.group_variables()
            
            self.hubo_total_cost.normalize()
            sample_like = {}
            for var in self.hubo_variables:
                sample_like[var] = 1
            max_cost = self.hubo_total_cost.energy(sample_like)   
                
            self.scaler = max_cost
            #print("Scaler: ", self.scaler)
            
            if self.query_graph_name == "clique":
                self.every_rank_appears_exactly_once()
                self.construct_validity_constraints_3()
                self.construct_validity_constraints_4()
            elif self.query_graph_name == "tree" and not is_chain_tree(self.query_graph) and not is_star_graph(self.query_graph):
                self.at_every_rank_select_rank_many_joins()
                self.select_same_join_for_proceeding_ranks()
                self.respect_query_graph2()
            else:
                self.at_every_rank_select_rank_many_joins()
                self.select_same_join_for_proceeding_ranks()
                self.respect_query_graph()
        
        self.method_name = method_name
        self.construct_full_hubo()
        if create_bqm:
            self.construct_BQM(hubo_to_bqm_strength)
            print("Number of terms in QUBO: ", len(self.bqm.linear) + len(self.bqm.quadratic))
    
    
    def at_every_rank_select_rank_many_joins(self):
        scaler = 10*self.scaler
        for rank in self.ranks:
            if len(self.variables_by_rank[rank]) > rank:
                bqm = dimod.generators.combinations(self.variables_by_rank[rank], rank + 1, strength = scaler)
            else:
                bqm = dimod.generators.combinations(self.variables_by_rank[rank], len(self.variables_by_rank[rank]), strength = scaler)
            for bvar in bqm.linear:
                self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
            for bvar in bqm.quadratic:
                self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
                
                
    def select_same_join_for_proceeding_ranks(self):
        for join in self.variables_by_joins:
            for rank in list(sorted(list(self.ranks)))[1:]:
                var1 = ((join[0], join[1], rank - 1),)
                self.safe_append(self.validity_constraints, var1, self.scaler, mode="int")
                var2 = ((join[0], join[1], rank - 1), (join[0], join[1], rank))
                self.safe_append(self.validity_constraints, var2, -self.scaler, mode="int")
          
                
    def respect_query_graph(self):
        scaler = self.scaler
        for rank in self.ranks:
            for join1 in self.variables_by_joins:
                for join2 in self.variables_by_joins:
                    if join1[0] not in join2 or join1[1] not in join2:
                        if join1[0] in join2 or join1[1] in join2:
                            var1 = ((join1[0], join1[1], rank), (join2[0], join2[1], rank))
                            self.safe_append(self.validity_constraints, var1, -scaler, mode="int")
                            
                    
    def respect_query_graph2(self):
        scaler = self.scaler
        for rank in self.ranks:
            if rank == 0:
                continue
            positive_pairs = set()
            for join1 in self.variables_by_joins:
                for join2 in self.variables_by_joins:
                    if join1[0] not in join2 or join1[1] not in join2:
                        if join1[0] in join2 or join1[1] in join2:
                            var1 = ((join1[0], join1[1], rank), (join2[0], join2[1], rank))
                            var2 = ((join2[0], join2[1], rank), (join1[0], join1[1], rank))
                            if var1 not in positive_pairs and var2 not in positive_pairs:
                                positive_pairs.add(var1)

            
            if is_star_graph(self.query_graph):
                print("Star graph")
                terms = self.hubo_combinations2(positive_pairs, ncr(rank + 1, 2), scaler)
            elif self.query_graph_name == "clique":
                print("Clique graph")
                terms = self.hubo_combinations2(positive_pairs, ncr(rank + 1, 2), scaler)
            else:
                print("Not a star graph")
                terms = self.hubo_combinations2(positive_pairs, rank, scaler)
            for term in terms:
                self.safe_append(self.validity_constraints, term, terms[term], mode="int")
        
    
    def draw_query_graph(self, name="query_graph.png"):
        node_color = 'sandybrown'
        node_size = 600
        edge_color = 'gray'
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('whitesmoke')

        nx.draw(self.query_graph,
                pos=nx.spring_layout(self.query_graph, seed=42),  # Use a spring layout for positioning
                node_color=node_color,
                node_size=node_size,
                edge_color=edge_color,
                with_labels=True,  # Display node labels
                font_size=10,      # Set font size for labels
                font_color='black',# Set font color for labels
                width=3.0,         # Set edge width
                style='solid',     # Set edge style
                alpha=1.0          # Set transparency
            )

        plt.savefig(name)
    
    
    def safe_append(self, dictionary, key, value, mode="list"):
        if key in dictionary:
            if mode == "list":
                dictionary[key].append(value)
            elif mode == "int":
                dictionary[key] = dictionary[key] + value
        else:
            if mode == "list":
                dictionary[key] = [value]
            elif mode == "int":
                dictionary[key] = value
    
    
    def group_variables(self):
        self.variables_by_rank = {}
        for var in self.hubo_variables:
            rank = var[-1]
            self.safe_append(self.variables_by_rank, rank, var)    
                    
        self.variables_by_joins = {}
        for var in self.hubo_variables:
            join = (var[0], var[1])
            self.safe_append(self.variables_by_joins, join, var)
                
        self.variables_by_tables = {}
        for var in self.hubo_variables:
            for table in var[0:2]:
                self.safe_append(self.variables_by_tables, table, var)
    
    
    def construct_presice_cost_function(self):
        for rank in self.ranks:
            joined_rels = []
            if rank > 0:
                for var in self.variables:
                    if self.variables[var][-1].get_rank() == rank - 1:
                        joined_rels.append(var)  
            for edge in self.query_graph.edges(data=True):
                rel1, rel2 = edge[0], edge[1]
                if rank == 0:
                    subgraph = frozenset([rel1, rel2])
                    self.variables[subgraph] = [Variable(self.relations, self.selectivities, subgraph, rel1, rel2, rank, [])]
                else:
                    for joined in joined_rels:
                        if (rel1 in joined and rel2 not in joined) or (rel1 not in joined and rel2 in joined):
                            new_joined = joined.union(frozenset([rel1, rel2]))
                            added_variables = []
                            for var in self.variables[joined]:
                                added_variables.append(Variable(self.relations, self.selectivities, new_joined, rel1, rel2, rank, var.get_labeling()))
                            if new_joined in self.variables:
                                self.variables[new_joined].extend(added_variables)
                            else:
                                self.variables[new_joined] = added_variables      
        
        for v in self.variables:
            for var in self.variables[v]:
                labeling = var.get_labeling()
                cost = var.get_local_cost()
                self.variables_dict[tuple(labeling)] = cost
                
        self.hubo = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo_total_cost = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo.normalize()
        self.hubo_variables = self.hubo.variables
        print("Number of variables: ", len(self.hubo_variables))
        print("Number of terms in HUBO: ", len(self.hubo))
        #for term in self.hubo:
        #    print(term, self.hubo[term])
        self.normalized_variable_dict, off = self.hubo.to_hubo()
    
    
    def construct_estimate_cost_function(self):
        for rank in self.ranks:
            
            # Select the sub-plans with the minimum cost from the previous rank
            # similar to the greedy approach
            # ----------------------------------------------
            min_keys = [(None, math.inf) for _ in range(self.estimation_size)]
            if rank > 0:
                for var in self.variables:
                    for v in self.variables[var]:
                        if v.get_rank() == rank - 1:
                            last_ranks_min = v.get_local_cost()
                            second_terms = [key[1] for key in min_keys]
                            if last_ranks_min < max(second_terms) and last_ranks_min not in second_terms:
                                index = second_terms.index(max(second_terms))
                                min_keys[index] = (var, last_ranks_min)                   
            min_keys = [min_key[0] for min_key in min_keys if min_key[0] is not None]
            #print("Min keys: ", min_keys)
            # ----------------------------------------------
            
            for edge in self.query_graph.edges(data=True):
                rel1, rel2 = edge[0], edge[1]
                if rank == 0:
                    subgraph = frozenset([edge[0], edge[1]])
                    self.variables[subgraph] = [Variable(self.relations, self.selectivities, subgraph, edge[0], edge[1], rank, [])]
                else:
                    for min_key in min_keys:
                        if (rel1 in min_key and rel2 not in min_key) or (rel1 not in min_key and rel2 in min_key):
                            new_subgraph = frozenset(tuple(min_key) + (rel1, rel2))
                            added_subgraphs = []
                            
                            for vv in self.variables[min_key]:
                                labeling = vv.get_labeling()
                                new_var = Variable(self.relations, self.selectivities, new_subgraph, rel1, rel2, rank, labeling)
                                added_subgraphs.append(new_var)
                            
                            for vvv in added_subgraphs:
                                if new_subgraph in self.variables:
                                    self.variables[new_subgraph].append(vvv)
                                else:
                                    self.variables[new_subgraph] = [vvv]
                                        
        for v in self.variables:
            for var in self.variables[v]:
                labeling = var.get_labeling()
                cost = var.get_local_cost()
                self.variables_dict[tuple(labeling)] = cost
        
        if self.query_graph_name == "clique":
            variables_by_length = {}
            for l in range(1, len(self.query_graph.nodes)):
                variables_by_length[l] = []
                for var in self.variables_dict:
                    if len(var) == l:
                        variables_by_length[l].append((var, self.variables_dict[var]))
            
            new_variables_dict = {}
            for l in variables_by_length:
                if l == 1:
                    min_elem = min(variables_by_length[l], key=lambda x: x[1])
                    new_variables_dict[min_elem[0]] = min_elem[1]
                else:
                    prev_var = [key for key in new_variables_dict.keys() if len(key) == l - 1][0]
                    # Find the min element so that prev_var is a subset of it
                    min_elem = None
                    for elem in variables_by_length[l]:
                        if set(prev_var).issubset(set(elem[0])):
                            if min_elem is None:
                                min_elem = elem
                            if elem[1] < min_elem[1]:
                                min_elem = elem
                    new_variables_dict[min_elem[0]] = min_elem[1]
            self.variables_dict = new_variables_dict
            
        # Due to limiting number of variables in the previous step,
        # there are some redundant variables that are not needed for the final solution
        # ---------------------------------------------------------------
        labelings_for_full_join = [key for key in self.variables_dict.keys() if len(key) == len(self.query_graph.nodes) - 1]
        #print(labelings_for_full_join)
        del_keys = []
        for e in self.variables_dict:
            is_subset = False
            if len(e) == len(self.query_graph.nodes) - 1:
                continue
            for labeling in labelings_for_full_join:
                if set(e).issubset(set(labeling)):
                    is_subset = True
            if not is_subset:
                del_keys.append(e)
        
        for key in del_keys:
            del self.variables_dict[key]
        # ---------------------------------------------------------------
        print("variables dict at this point", self.variables_dict)
        self.hubo = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo_total_cost = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo.normalize()
        self.hubo_variables = self.hubo.variables
        print("Number of variables: ", len(self.hubo_variables))
        print("Number of terms in HUBO: ", len(self.hubo))
        self.normalized_variable_dict, off = self.hubo.to_hubo()
    
    
    def every_rank_appears_exactly_once(self):
        scaler = self.scaler
        for l in self.ranks:
            vars = self.variables_by_rank[l]
            bqm = dimod.generators.combinations(vars, 1, strength = scaler)
            for bvar in bqm.linear:
                self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
            for bvar in bqm.quadratic:
                self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
            self.offset = self.offset + bqm.offset


    # Encode (1 - labelings for variables[frozenset({0, 1, 2, 3, 4})])^2
    # (1 - x - y - z)^2 = x^2 + 2 x y + 2 x z - 2 x + y^2 + 2 y z - 2 y + z^2 - 2 z + 1
    # = -x - y - z + 2 x y + 2 x z + 2 y z + 1
    # This is the same function as dimod.generators.combinations but for higher-order models
    def hubo_combinations(self):
        scaler = self.scaler
        labelings_for_full_join = [key for key in self.variables_dict.keys() if len(key) == len(self.query_graph.nodes) - 1] #[var.get_labeling() for var in self.variables[frozenset(self.query_graph.nodes)]]
        for labeling in labelings_for_full_join:
            if tuple(labeling) in self.validity_constraints:
                self.validity_constraints[tuple(labeling)] = self.validity_constraints[tuple(labeling)] - scaler
            else:
                self.validity_constraints[tuple(labeling)] = -scaler
        
        print(labelings_for_full_join)
        print("Creating combinations...", len(labelings_for_full_join))
        for comb in combinations(labelings_for_full_join, 2):
            if comb[0] != comb[1]:
                prod = tuple(comb[0] + comb[1])
                if prod in self.validity_constraints:
                    self.validity_constraints[prod] = self.validity_constraints[prod] + 2*scaler
                else:
                    self.validity_constraints[prod] = 2*scaler
    
    
    # Encode (1 - labelings for variables[frozenset({0, 1, 2, 3, 4})])^2
    # (k - x - y - z)^2 = x^2 + 2 x y + 2 x z - 2 x + y^2 + 2 y z - 2 y + z^2 - 2 z + 1
    # = -2kx + x - 2ky + y - 2kz + z + 2 x y + 2 x z + 2 y z + 1
    # This is the same function as dimod.generators.combinations but for higher-order models           
    def hubo_combinations2(self, variables, k, scaler = 1):
        result = {}
        for var in variables:
            #print(var)
            result[tuple(var)] = (-2*k* + 1)*scaler**2
        
        print("Creating combinations...", len(variables))
        for comb in combinations(variables, 2):
            if comb[0] != comb[1]:
                prod = tuple(comb[0] + comb[1])
                #print(prod)
                result[prod] = 2*scaler**2
        
        return result
        
                
                
    # select exactly n_joins many different joins            
    def construct_validity_constraints_2(self):
        scaler = self.scaler
        aux_vars = []
        for x, vars in self.variables_by_joins.items():
            aux_var = 'aux_' + str(x)
            aux_vars.append(aux_var)
            combs = combinations_with_variable(aux_var, vars, scaler=scaler)
            for comb in combs:
                self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
        # Select exactly n_joins many aux_vars to be true
        bqm = dimod.generators.combinations(aux_vars, self.max_number_of_ranks, strength=scaler)
        for bvar in bqm.linear:
            self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
        for bvar in bqm.quadratic:
            self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
    
    
    def every_join_includes_exactly_one_new_and_one_old_table(self):
        scaler = 2*self.scaler
        tables_by_ranks = {}
        variables_by_tables = {}
        for rank in self.ranks:
            tables_by_ranks[rank] = []
            for table in self.relations:
                tables_by_ranks[rank].append((table, rank))
                if table in variables_by_tables:
                    variables_by_tables[table].append((table, rank))
                else:
                    variables_by_tables[table] = [(table, rank)]
        
        for rank in list(self.ranks):
            for table in self.relations:
                table_rank_vars = [(table, r) for r in range(rank + 1, self.max_number_of_ranks)]
                if len(table_rank_vars) > 0:
                    v = (table, rank)
                    #print("If ", v, " is true then all the following variables should be true: ", table_rank_vars)
                    combs = combinations_with_variable(v, table_rank_vars, scaler=scaler, coeff=len(table_rank_vars))
                    for comb in combs:
                        self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
        
        for join in self.variables_by_joins:
            join_vars = self.variables_by_joins[join]
            for var in join_vars:
                table1, table2 = var[0], var[1]
                rank = var[2]
                if rank > 0:
                    new_vars = [(table1, rank - 1), (table2, rank - 1)]
                    print("If ", var, " is true, then one of the following variables should be true too: ", new_vars)
                    combs = combinations_with_variable(var, new_vars, scaler=scaler)
                    for comb in combs:
                        self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
        
        # Select exactly two tables to be true at rank 0
        varss = [(table, 0) for table in self.relations]
        bqm = dimod.generators.combinations(varss, 2, strength=scaler)
        for bvar in bqm.linear:
            self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
        for bvar in bqm.quadratic:
            self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
            
            
    def every_join_performed_at_most_once(self):
        scaler = self.scaler
        aux_vars = []
        for join in self.variables_by_joins:
            join_vars = self.variables_by_joins[join]
            aux_var = 'au_' + str(join)
            aux_vars.append(aux_var)
            combs = combinations_with_variable(aux_var, join_vars, scaler=scaler)
            for comb in combs:
                self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
        
        # Select exactly n_joins many aux_vars to be true
        bqm = dimod.generators.combinations(aux_vars, self.max_number_of_ranks, strength=scaler)
        for bvar in bqm.linear:
            self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
        for bvar in bqm.quadratic:
            self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
        self.offset = self.offset + bqm.offset
                
    
    def every_table_appears_once_or_twice(self):
        scaler = self.scaler
        new_bqm = dict()
        aux_vars = []
        for table_id, tables in self.variables_by_tables.items():
            #print(table_id, tables)
            aux_var = 'aux_' + str(table_id)
            aux_vars.append(aux_var)
            # Linear terms
            new_bqm[(aux_var, )] = 3 * scaler
            for table in tables:
                new_bqm[(table,)] = -scaler
            # Quadratic terms
            for table in tables:
                new_bqm[(aux_var, table)] = -2 * scaler
            for comb in combinations(tables, 2):
                new_bqm[comb] = 2 * scaler
        
        for key in new_bqm:
            self.safe_append(self.validity_constraints, key, new_bqm[key], mode="int")
            
        # Select exactly n_joins - 1 many aux_vars to be true
        bqm = dimod.generators.combinations(aux_vars, self.max_number_of_ranks - 1, strength=scaler)
        for bvar in bqm.linear:
            self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
        for bvar in bqm.quadratic:
            self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
        
          
                
    # rank + 1 should be connected to rank so that if (x, y, l) and (x', y', l + 1) 
    # and x != x' and y != y' then (x, y, l) and (x', y', l + 1) should be penalized
    def construct_validity_constraints_3(self):
        scaler = self.scaler
        for rank in self.ranks[1:]:
            join_vars = self.variables_by_rank[rank]
            prev_join_vars = self.variables_by_rank[rank - 1]
            for var1 in join_vars:
                for var2 in prev_join_vars:
                    if (var1[0] == var2[0] and var1[1] == var2[1]) or (var1[0] != var2[0] and var1[1] != var2[1] and var1[0] != var2[1] and var1[1] != var2[0]):
                        self.safe_append(self.validity_constraints, (var1, var2), scaler, mode="int")
    
    
    # Every table has to appear at least once (max as many times as the number of joins)
    def construct_validity_constraints_4(self):
        scaler = 1
        for table_id, tables in self.variables_by_tables.items():
            terms = table_number_constraint(self.max_number_of_ranks, tables, table_id)
            linear = {k[0]: v for k, v in terms.items() if len(k) == 1}  # Single variable tuples for linear terms
            quadratic = {k: v for k, v in terms.items() if len(k) == 2}  # Pairs of variables for quadratic terms
            bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.Vartype.BINARY)
            bqm.scale(scaler)
            for bvar in bqm.linear:
                self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
            for bvar in bqm.quadratic:
                self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
    
    
    def construct_full_hubo(self):
        full_variable_dict = self.normalized_variable_dict.copy()
        #full_variable_dict = dict()
        for var in self.validity_constraints:
            self.safe_append(full_variable_dict, var, self.validity_constraints[var], mode="int")
        self.full_hubo = dimod.BinaryPolynomial(full_variable_dict, dimod.Vartype.BINARY)
        #self.full_hubo.normalize()           
    
    
    def construct_BQM(self, strength):
        bqm_hubo = self.full_hubo.copy()
        problem_dict, off = bqm_hubo.to_hubo()
        self.bqm = dimod.make_quadratic(problem_dict, strength = strength, vartype = dimod.Vartype.BINARY)
        self.bqm.offset = self.offset
    
    
    def solve_with_exact_poly_solver(self):
        solver = dimod.ExactPolySolver()
        print("Number of variables in final objective: ", len(self.full_hubo.variables))
        result = solver.sample_poly(self.full_hubo)
        print("Energy: ", result.first.energy + self.offset)
        self.samplesets["exact_poly_solver"] = result
        return result
    
    def solve_with_exact_BQM_solver(self):
        solver = dimod.ExactSolver()
        result = solver.sample(self.bqm)
        self.samplesets["exact_BQM_solver"] = result
        return result
    
    
    def solve_with_TabuSampler(self, num_reads=1000):
        sampler = TabuSampler()
        result = sampler.sample(self.bqm, num_reads=num_reads)
        self.samplesets["tabu"] = result
        return result
    
    
    def solve_with_simulated_annealing(self):
        sampler = SimulatedAnnealingSampler()
        result = sampler.sample(self.bqm, num_reads=10000)
        self.samplesets["simulated_annealing"] = result
        return result
    
    
    def solve_with_steepest_descent(self):
        sampler = SteepestDescentSampler()
        result = sampler.sample(self.bqm)
        self.samplesets["steepest_descent"] = result
        return result
    
    
    def solve_with_dynamic_programming(self):
        result = dynamic_programming(self.relations, self.selectivities)
        cost = basic_cost(result, self.relations, self.selectivities)
        self.samplesets["dynamic_programming"] = { "result": result, "cost": cost }
        return result, cost
    
    
    def evaluate_cost(self, result):
        cost = self.hubo_total_cost.energy(result)
        return cost
    
    
    def qubo_to_lp(self):
        cplex_problem = cplex.Cplex()
        cplex_problem.objective.set_sense(cplex_problem.objective.sense.minimize)
        variable_symbols = [str(var) for var in self.bqm.variables]
        cplex_problem.variables.add(names=variable_symbols, 
                                            types=[cplex_problem.variables.type.binary]*len(variable_symbols))

        linear_coeffs = self.bqm.linear
        obj_list = [(str(name), coeff) for name, coeff in linear_coeffs.items()]
        cplex_problem.objective.set_linear(obj_list)

        quadratic_coeffs = self.bqm.quadratic
        obj_list = [(str(name[0]), str(name[1]), coeff) for name, coeff in quadratic_coeffs.items()]
        cplex_problem.objective.set_quadratic_coefficients(obj_list)
        cplex_problem.write(self.lp_file)
        return cplex_problem
    
    
    def solve_with_CPLEX(self, print_log = False):
        cplex_problem = self.qubo_to_lp()
        
        if print_log:
            #cplex_problem.set_log_stream(None)
            cplex_problem.set_error_stream(None)
            cplex_problem.set_warning_stream(None)
            #cplex_problem.set_results_stream(None)
        
        cplex_result_file = open(self.name + "_cplex.log", "w")
        cplex_problem.set_results_stream(cplex_result_file)
        cplex_problem.set_log_stream(cplex_result_file)
        
        # Allow multiple threads
        cplex_problem.parameters.threads.set(8)
        # Print progress
        #cplex_problem.parameters.mip.display.set(2)
        # Set time limit
        #cplex_problem.parameters.timelimit.set(30)
        
        time_start = time.time()
        cplex_problem.solve()
        time_end = time.time()
        elapsed_time = time_end - time_start
        status = cplex_problem.solution.get_status_string()
        result = cplex_problem.solution.get_values()
        variables = cplex_problem.variables.get_names()
        value = cplex_problem.solution.get_objective_value() + self.bqm.offset
        self.samplesets["cplex"] = { "status": status, "energy": value, 
                                    "time": elapsed_time, "result": dict(zip(variables, result)) }
        return self.samplesets["cplex"]
    
    
    def solve_with_Gurobi(self, print_log = False):
        cplex_problem = self.qubo_to_lp()
        with gp.Env(empty=True) as env:  # type: ignore
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.read(self.lp_file, env) as model:  # type: ignore
                if not print_log:
                    model.Params.LogToConsole = 0
                model.Params.LogFile = "logs//" + self.name +  "_gurobi.log"
                model.Params.OutputFlag = 1
                model.Params.MIPFocus = 0 # aims to find a single optimal solution
                #model.Params.PoolSearchMode = 0 # No need for multiple solutions
                model.Params.PoolGap = 0.0 # Only provably optimal solutions are added to the pool
                model.Params.TimeLimit = 60
                model.Params.NumericFocus = 3
                #model.Params.Threads = 8
                model.presolve() # Decreases quality of solutions
                time_start = time.time()
                model.optimize()
                time_end = time.time()
                elapsed_time = time_end - time_start
                result = model.getAttr("X", model.getVars())
                variables = [var.VarName for var in model.getVars()]
                variables = [var.split("#")[0] for var in variables]
                status = model.Status
                min_energy = model.ObjVal + self.bqm.offset

        self.samplesets["gurobi"] = { "result": dict(zip(variables, result)), 
                                        "time": elapsed_time, "energy": min_energy, 
                                        "status": status }
        return self.samplesets["gurobi"]
    
    
    def solve_with_qaoa_pennylane(self):
        qaoa = QuantumApproximateOptimizationAlgorithm(self.full_hubo, {})
        qaoa.solve_with_QAOA_pennylane()
        return qaoa.samplesets["qaoa_pennylane"]
    
    def solve_with_qaoa_qiskit(self):
        qaoa = QuantumApproximateOptimizationAlgorithm(self.full_hubo, {})
        qaoa.solve_with_QAOA_Qiskit2()
        return qaoa.samplesets["qaoa_qiskit"]
    
    
    def solve_with_LeapHybridSampler(self):
        sampler = LeapHybridSampler()
        result = sampler.sample(self.bqm)
        self.samplesets["leap_hybrid"] = result
        return result
    
    
    def solve_with_DWave_Sampler(self):
        sampler = EmbeddingComposite(DWaveSampler())
        result = sampler.sample(self.bqm)
        self.samplesets["dwave"] = result
        return result
    
    
    def solve_with_KerberosSampler(self):
        sampler = KerberosSampler()
        result = sampler.sample(self.bqm)
        self.samplesets["kerberos"] = result
        return result
    
    def solve_with_greedy(self):
        result = greedy(self.query_graph, self.relations, self.selectivities)
        cost = basic_cost(result, self.relations, self.selectivities)
        self.samplesets["greedy"] = { "result": result, "cost": cost }
        return result, cost
    
    def solve_with_greedy_with_query_graph(self):
        result = greedy_with_query_graph(self.query_graph, self.relations, self.selectivities)
        cost = basic_cost(result, self.relations, self.selectivities)
        self.samplesets["greedy_with_query_graph"] = { "result": result, "cost": cost }
        return result, cost
    
    def solve_with_graph_aware_dynamic_programming(self):
        result = graph_aware_dynamic_programming(self.query_graph, self.relations, self.selectivities)
        cost = basic_cost(result, self.relations, self.selectivities)
        self.samplesets["graph_aware_dynamic_programming"] = { "result": result, "cost": cost }
        return result, cost
    
    
    def get_number_of_hubo_variables(self):
        return len(self.full_hubo.variables)
    
    def get_hubo_variables(self):
        return self.full_hubo.variables
    
    def get_number_of_bqm_variables(self):
        return len(self.bqm.variables)
    
    def get_number_of_hubo_terms(self):
        return len(self.full_hubo)
    
    def get_number_of_bqm_terms(self):
        return len(self.bqm.linear) + len(self.bqm.quadratic)
    
    def get_estimation_size(self):
        return self.estimation_size
    
    def get_cost_hubo(self):
        return self.hubo_total_cost
    
    def get_validity_bqm(self):
        full_variable_dict = dict()
        for var in self.validity_constraints:
            self.safe_append(full_variable_dict, var, self.validity_constraints[var], mode="int")
        full_hubo = dimod.BinaryPolynomial(full_variable_dict, dimod.Vartype.BINARY)
        return full_hubo
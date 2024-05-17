from itertools import combinations
import time
import cplex
import dimod
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from dwave.system import LeapHybridSampler
from hybrid.reference import KerberosSampler
from dwave.samplers import TabuSampler, SteepestDescentSolver, SimulatedAnnealingSampler, TreeDecompositionSolver, SteepestDescentSampler

from classical_algorithms.dynamic_programming import dynamic_programming
from classical_algorithms.weights_costs import basic_cost
from variable import Variable

class QJoin:
    
    def __init__(self, query_graph, scaler=1):
        self.query_graph = query_graph
        self.scaler = scaler
        # For left-deep plan the max number of levels in the same as number of nodes - 1
        # For bushy plan the max number of levels is less but it's always the following value
        self.max_number_of_levels = len(query_graph.edges) - len(nx.cycle_basis(query_graph))
        
        self.levels = range(self.max_number_of_levels)

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
        self.variables_by_levels = {}
        self.variables_by_joins = {}
        self.validity_constraints = {}
        self.samplesets = {}
        
        self.construct_cost_function()
        self.group_variables()
        self.every_level_one_join()
        self.construct_validity_constraints_1()
        self.construct_full_hubo()
    
    
    def draw_query_graph(self):
        # Define node colors and sizes
        node_color = 'sandybrown'
        node_size = 600

        # Define edge colors
        edge_color = 'gray'

        # Set the background color of the plot
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('whitesmoke')

        nx.draw(self.query_graph,
                pos=nx.spring_layout(self.query_graph, seed=42),  # Use a spring layout for positioning
                node_color=node_color,
                node_size=node_size,
                edge_color=edge_color,
                with_labels=False,  # Display node labels
                font_size=10,      # Set font size for labels
                font_color='black',# Set font color for labels
                width=3.0,         # Set edge width
                style='solid',     # Set edge style
                alpha=1.0          # Set transparency
            )

        # Show the plot
        plt.show()
        
        
    def get_connected_subgraphs_dfs(self, graph, node, n):
        connected_subgraphs = set()
        
        # Function to perform DFS to find connected subgraphs of size n
        def dfs(start_node, path):
            
            if len(path) == n + 1:
                connected_subgraphs.add(tuple(path))
                return
            
            for r in range(1, n - len(path) + 2):
                for neighbors_combination in combinations(graph.neighbors(start_node), r):
                    for neighbor in neighbors_combination:
                        if neighbor not in path:
                            dfs(neighbor, path.union(set(neighbors_combination)))
        
        for r in range(1, n + 1):
            for neighbors_combination in combinations(graph.neighbors(node), r):
                for neighbor in neighbors_combination:
                    dfs(neighbor, set(list(neighbors_combination) + [node]))
        
        return connected_subgraphs
    
    
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
        self.variables_by_levels = {}
        for level in self.levels:
            self.variables_by_levels[level] = []
            for v in self.hubo_variables:
                if v[2] == level:
                    self.variables_by_levels[level].append(v)
                    
        self.variables_by_joins = {}
        for var in self.hubo_variables:
            join = (var[0], var[1])
            self.safe_append(self.variables_by_joins, join, var)
                
        self.variables_by_tables = {}
        for var in self.hubo_variables:
            for table in var[0:2]:
                self.safe_append(self.variables_by_tables, table, var)
    
    
    def construct_cost_function(self):
        for level in self.levels:
            for edge in self.query_graph.edges(data=True):
                
                if level == 0:
                    subgraph = frozenset([edge[0], edge[1]])
                    self.variables[subgraph] = [Variable(self.relations, self.selectivities, subgraph, subgraph, edge[0], edge[1], level, [])]
                else:
                    join1 = edge[0]
                    join2 = edge[1]
                    
                    query_graph_copy = self.query_graph.copy()
                    edge_to_remove = (join1, join2)
                    query_graph_copy.remove_edge(*edge_to_remove)
                    
                    connected_subgraphs = self.get_connected_subgraphs_dfs(query_graph_copy, join1, level)
                    connected_subgraphs = connected_subgraphs.union(self.get_connected_subgraphs_dfs(query_graph_copy, join2, level))
                    
                    for subgraph in connected_subgraphs:
                        new_subgraph = frozenset(subgraph + (join1, join2))
                        subgraph = frozenset(subgraph)
                        new_variables = {}
                        
                        for var in self.variables[subgraph]:
                            
                            if new_subgraph not in new_variables:
                                new_variables[new_subgraph] = []
                            
                            new_variables[new_subgraph].append(Variable(self.relations, self.selectivities, new_subgraph, subgraph, join1, join2, level, var.get_labeling()))
                        
                        for new_s in new_variables:
                            if new_s in self.variables:
                                self.variables[new_s].extend(new_variables[new_s])
                            else:
                                self.variables[new_s] = new_variables[new_s]
        
        for v in self.variables:
            for var in self.variables[v]:
                labeling = var.get_labeling()
                self.variables_dict[tuple(labeling)] = var.get_local_cost()
                
    
        self.hubo = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo_total_cost = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo.normalize()
        self.hubo_variables = self.hubo.variables
        #print("Number of variables: ", len(self.hubo_variables))
        self.normalized_variable_dict, off = self.hubo.to_hubo()


    # At every level we perform exactly one join
    def every_level_one_join(self):
        for l in self.variables_by_levels:
            vars = self.variables_by_levels[l]
            bqm = dimod.generators.combinations(vars, 1, strength = 1)
            
            for bvar in bqm.linear:
                self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
            
            for bvar in bqm.quadratic:
                self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
    
                 
    # (x - a - b - c)^2 = a^2 + 2 a b + 2 a c - 2 a x + b^2 + 2 b c - 2 b x + c^2 - 2 c x + x^2
    def combinations_with_variable(self, x, vars, scaler = 1):
        result = {}
        result[(x,)] = scaler
        for v in vars:
            result[(v,)] = scaler
            result[(x, v)] = -2 * scaler
        
        for comb in combinations(vars, 2):
            result[comb] = 2 * scaler
        
        return result
    
    # (1 + 2*x_2 + ... + max_number_of_levels * x_n - sum over table vars)^2
    def table_number_constraint(self, tables, table_id, scaler = 1):
        integer_vars = []
        N = self.max_number_of_levels
        M = int(np.floor(np.log2(N)))
        result = {}
        
        # x_0 = 1 is fixed
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
    
    def construct_validity_constraints_1(self):
        # Encode (1 - labelings for variables[frozenset({0, 1, 2, 3, 4})])^2
        # (1 - x - y - z)^2 = x^2 + 2 x y + 2 x z - 2 x + y^2 + 2 y z - 2 y + z^2 - 2 z + 1
        # = -x - y - z + 2 x y + 2 x z + 2 y z + 1    
        scaler = self.scaler
        labelings_for_full_join = [var.get_labeling() for var in self.variables[frozenset(self.query_graph.nodes)]]
        for labeling in labelings_for_full_join:
            
            if tuple(labeling) in self.validity_constraints:
                self.validity_constraints[tuple(labeling)] = self.validity_constraints[tuple(labeling)] - scaler
            else:
                self.validity_constraints[tuple(labeling)] = -scaler
                
        for comb in combinations(labelings_for_full_join, 2):
            prod = tuple(comb[0] + comb[1])
            
            if prod in self.validity_constraints:
                self.validity_constraints[prod] = self.validity_constraints[prod] + 2*scaler
            else:
                self.validity_constraints[prod] = 2*scaler
                
                
    def construct_validity_constraints_2(self):
        # select exactly n_joins many different joins
        aux_vars = []
        for x, vars in self.variables_by_joins.items():
            aux_var = 'aux_' + str(x)
            aux_vars.append(aux_var)
            combs = self.combinations_with_variable(aux_var, vars, scaler=3)
            for comb in combs:
                self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
        # Select exactly n_joins many aux_vars to be true
        bqm = dimod.generators.combinations(aux_vars, self.max_number_of_levels, strength=3)
        for bvar in bqm.linear:
            self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
        for bvar in bqm.quadratic:
            self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
                
                
    def construct_validity_constraints_3(self):
        # level + 1 should be connecte to level so that if (x, y, l) and (x', y', l + 1) and x != x' and y != y' then (x, y, l) and (x', y', l + 1) should be penalized
        scaler = 5
        for level in self.levels[1:]:
            join_vars = self.variables_by_levels[level]
            prev_join_vars = self.variables_by_levels[level - 1]
            for var1 in join_vars:
                for var2 in prev_join_vars:
                    if var1[0] != var2[0] and var1[1] != var2[1]:
                        # Penalize if two joins are not connected
                        self.safe_append(self.validity_constraints, (var1, var2), scaler, mode="int")
    
    def construct_validity_constraints_4(self):
        # Every table has to appear at least once (max as many times as the number of joins)
        for table_id, tables in self.variables_by_tables.items():
            combs = self.table_number_constraint(tables, table_id, scaler=1)
            for comb in combs:
                self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
    
    
    def construct_full_hubo(self):
        full_variable_dict = self.normalized_variable_dict.copy()
        for var in self.validity_constraints:
            self.safe_append(full_variable_dict, var, self.validity_constraints[var], mode="int")
        self.full_hubo = dimod.BinaryPolynomial(full_variable_dict, dimod.Vartype.BINARY)            
    
    
    def construct_BQM(self, strength=5):
        problem_dict, off = self.full_hubo.to_hubo()
        self.bqm = dimod.make_quadratic(problem_dict, strength = strength, vartype = dimod.Vartype.BINARY)
    
    
    def solve_with_exact_poly_solver(self):
        solver = dimod.ExactPolySolver()
        result = solver.sample_poly(self.full_hubo)
        self.samplesets["exact_poly_solver"] = result
        return result
    
    
    def solve_with_TabuSampler(self, num_reads=1000, strength=5):
        self.construct_BQM(strength)
        sampler = TabuSampler()
        result = sampler.sample(self.bqm, num_reads=num_reads)
        self.samplesets["tabu"] = result
        return result
    
    
    def solve_with_simulated_annealing(self):
        self.construct_BQM()
        sampler = SimulatedAnnealingSampler()
        result = sampler.sample(self.bqm)
        self.samplesets["simulated_annealing"] = result
        return result
    
    
    def solve_with_steepest_descent(self):
        self.construct_BQM()
        sampler = SteepestDescentSampler()
        result = sampler.sample(self.bqm)
        self.samplesets["steepest_descent"] = result
        return result
    
    
    def solve_with_dynamic_programming(self):
        result = dynamic_programming(self.query_graph, self.relations, self.selectivities)
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
                model.Params.LogFile = self.name +  "_gurobi.log"
                model.Params.OutputFlag = 1
                model.Params.LogLevel = 1
                model.Params.MIPFocus = 0 # aims to find a single optimal solution
                model.Params.PoolSearchMode = 0 # No need for multiple solutions
                model.Params.PoolGap = 0.0 # Only provably optimal solutions are added to the pool
                #model.Params.TimeLimit = 30
                #model.Params.Threads = 8
                #model.presolve() # Decreases quality of solutions
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
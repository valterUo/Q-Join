import copy
from itertools import combinations
import math
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
from qaoa.qaoa import QuantumApproximateOptimizationAlgorithm
from utils import combinations_with_variable, get_connected_subgraphs_with_dfs, table_number_constraint, Variable

class QJoin:
    
    def __init__(self, query_graph, scaler=1, hubo_to_bqm_strength=5, approximation = False):
        self.query_graph = query_graph
        self.scaler = scaler
        # For left-deep plan the max number of levels in the same as number of nodes - 1
        # For bushy plan the max number of levels is less but it's always the following value
        self.max_number_of_levels = len(query_graph.nodes) - 1 #len(nx.cycle_basis(query_graph))
        
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
        self.lp_file = "lp_files//" + str(query_graph) + "_qubo.lp"
        self.name = str(query_graph)
        
        if approximation:
            self.construct_estimate_cost_function()
            self.group_variables()
            # Hard constraints
            self.every_level_has_one_join()
            self.hubo_combinations()
            
            # Hard constraints
            #self.every_level_has_one_join()
            #self.construct_validity_constraints_2()
            #self.construct_validity_constraints_3()
            #self.construct_validity_constraints_4()
            #print("Hard constraints constructed")
        else:
            start = time.time()
            self.construct_presice_cost_function()
            end = time.time()
            print("Time for constructing precise cost function: ", end - start)
            self.group_variables()
            # Hard constraints
            self.every_level_has_one_join()
            self.hubo_combinations()
            
        self.construct_full_hubo()
        self.construct_BQM(strength=hubo_to_bqm_strength)
        print("Number of terms in QUBO: ", len(self.bqm.linear) + len(self.bqm.quadratic))
            
    
    
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
                with_labels=False,  # Display node labels
                font_size=10,      # Set font size for labels
                font_color='black',# Set font color for labels
                width=3.0,         # Set edge width
                style='solid',     # Set edge style
                alpha=1.0          # Set transparency
            )

        plt.savefig("results//" + name)
    
    
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
    
    
    def construct_presice_cost_function(self):
        for level in self.levels:
            
            joined_rels = []
            if level > 0:
                for var in self.variables:
                    if self.variables[var][-1].get_level() == level - 1:
                        joined_rels.append(var)
                        
            for edge in self.query_graph.edges(data=True):
                rel1, rel2 = edge[0], edge[1]
                
                if level == 0:
                    subgraph = frozenset([rel1, rel2])
                    self.variables[subgraph] = [Variable(self.relations, self.selectivities, subgraph, rel1, rel2, level, [])]
                else:
                    for joined in joined_rels:
                        
                        if (rel1 in joined and rel2 not in joined) or (rel1 not in joined and rel2 in joined):
                            
                            new_joined = joined.union(frozenset([rel1, rel2]))
                            added_variables = []
                            
                            for var in self.variables[joined]:
                                added_variables.append(Variable(self.relations, self.selectivities, new_joined, rel1, rel2, level, var.get_labeling()))
                                
                            if new_joined in self.variables:
                                self.variables[new_joined].extend(added_variables)
                            else:
                                self.variables[new_joined] = added_variables      
        
        for v in self.variables:
            for var in self.variables[v]:
                labeling = var.get_labeling()
                self.variables_dict[tuple(labeling)] = var.get_local_cost()
                
        #for e in self.variables_dict:
        #    print(e, self.variables_dict[e])
                
        self.hubo = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo_total_cost = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo.normalize()
        self.hubo_variables = self.hubo.variables
        print("Number of variables: ", len(self.hubo_variables))
        print("Number of terms in HUBO: ", len(self.hubo))
        self.normalized_variable_dict, off = self.hubo.to_hubo()
    
    
    def construct_estimate_cost_function(self, estimation_size = 1000):
        for level in self.levels:
            
            last_levels_mins = [math.inf for _ in range(estimation_size)]
            min_keys = [None for _ in range(estimation_size)]
            
            if level > 0:
                for var in self.variables:
                    if self.variables[var][-1].get_level() == level - 1:
                        var_with_level = self.variables[var][-1]
                        last_levels_min = var_with_level.get_local_cost()
                        
                        if last_levels_min < max(last_levels_mins):
                            index = last_levels_mins.index(max(last_levels_mins))
                            last_levels_mins[index] = last_levels_min
                            min_keys[index] = var
            #else:
            #    for edge in self.query_graph.edges(data=True):
            #        join1, join2 = edge[0], edge[1]
            #        cost = self.relations[join1]["cardinality"] * self.relations[join2]["cardinality"] * self.selectivities[(join1, join2)]["selectivity"]
            #        if cost < max(last_levels_mins):
            #            index = last_levels_mins.index(max(last_levels_mins))
            #            last_levels_mins[index] = cost
            #            min_keys[index] = frozenset([join1, join2])
                            
            min_keys = [min_key for min_key in min_keys if min_key is not None]
            
            for edge in self.query_graph.edges(data=True):
                
                join1, join2 = edge[0], edge[1]
                
                if level == 0:
                    #for min_key in min_keys:
                    subgraph = frozenset([edge[0], edge[1]])
                    self.variables[subgraph] = [Variable(self.relations, self.selectivities, subgraph, edge[0], edge[1], level, [])]
                else:
                    for min_key in min_keys:
                        if (join1 in min_key and join2 not in min_key) or (join1 not in min_key and join2 in min_key):
                            new_subgraph = frozenset(tuple(min_key) + (join1, join2))
                            labeling = self.variables[min_key][-1].get_labeling()
                            if new_subgraph in self.variables:
                                self.variables[new_subgraph].append(Variable(self.relations, self.selectivities, new_subgraph, join1, join2, level, labeling))
                            else:
                                self.variables[new_subgraph] = [Variable(self.relations, self.selectivities, new_subgraph, join1, join2, level, labeling)]
                      
        for v in self.variables:
            for var in self.variables[v]:
                labeling = var.get_labeling()
                self.variables_dict[tuple(labeling)] = var.get_local_cost()
                
        #for e in self.variables_dict:
        #    print(e, self.variables_dict[e])
        
        self.hubo = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo_total_cost = dimod.BinaryPolynomial(self.variables_dict, dimod.Vartype.BINARY)
        self.hubo.normalize()
        self.hubo_variables = self.hubo.variables
        print("Number of variables: ", len(self.hubo_variables))
        print("Number of terms in HUBO: ", len(self.hubo))
        self.normalized_variable_dict, off = self.hubo.to_hubo()


    # At every level we perform exactly one join
    def every_level_has_one_join(self, relative_scaler=1):
        scaler = relative_scaler*self.scaler
        for l in self.variables_by_levels:
            vars = self.variables_by_levels[l]
            if len(vars) > 0:
                bqm = dimod.generators.combinations(vars, 1, strength = scaler)
                for bvar in bqm.linear:
                    self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
                for bvar in bqm.quadratic:
                    self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
            else:
                print("No variables at level ", l)


    # Encode (1 - labelings for variables[frozenset({0, 1, 2, 3, 4})])^2
    # (1 - x - y - z)^2 = x^2 + 2 x y + 2 x z - 2 x + y^2 + 2 y z - 2 y + z^2 - 2 z + 1
    # = -x - y - z + 2 x y + 2 x z + 2 y z + 1
    # This is the same function as dimod.generators.combinations but for higher-order models
    def hubo_combinations(self):
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
                
                
    # select exactly n_joins many different joins            
    def construct_validity_constraints_2(self, relative_scaler=1):
        scaler = relative_scaler*self.scaler
        aux_vars = []
        for x, vars in self.variables_by_joins.items():
            aux_var = 'aux_' + str(x)
            aux_vars.append(aux_var)
            combs = combinations_with_variable(aux_var, vars, scaler=scaler)
            for comb in combs:
                self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
        # Select exactly n_joins many aux_vars to be true
        bqm = dimod.generators.combinations(aux_vars, self.max_number_of_levels, strength=scaler)
        for bvar in bqm.linear:
            self.safe_append(self.validity_constraints, (bvar,), bqm.linear[bvar], mode="int")
        for bvar in bqm.quadratic:
            self.safe_append(self.validity_constraints, bvar, bqm.quadratic[bvar], mode="int")
          
                
    # level + 1 should be connected to level so that if (x, y, l) and (x', y', l + 1) 
    # and x != x' and y != y' then (x, y, l) and (x', y', l + 1) should be penalized
    def construct_validity_constraints_3(self, relative_scaler=1):
        scaler = relative_scaler*self.scaler
        for level in self.levels[1:]:
            join_vars = self.variables_by_levels[level]
            prev_join_vars = self.variables_by_levels[level - 1]
            for var1 in join_vars:
                for var2 in prev_join_vars:
                    if var1[0] != var2[0] and var1[1] != var2[1]:
                        # Penalize if two joins are not connected
                        self.safe_append(self.validity_constraints, (var1, var2), scaler, mode="int")
    
    
    # Every table has to appear at least once (max as many times as the number of joins)
    def construct_validity_constraints_4(self, relative_scaler=1):
        scaler = relative_scaler*self.scaler
        for table_id, tables in self.variables_by_tables.items():
            combs = table_number_constraint(self.max_number_of_levels, tables, table_id, scaler=scaler)
            for comb in combs:
                self.safe_append(self.validity_constraints, comb, combs[comb], mode="int")
    
    
    def construct_full_hubo(self):
        full_variable_dict = self.normalized_variable_dict.copy()
        for var in self.validity_constraints:
            self.safe_append(full_variable_dict, var, self.validity_constraints[var], mode="int")
        self.full_hubo = dimod.BinaryPolynomial(full_variable_dict, dimod.Vartype.BINARY)            
    
    
    def construct_BQM(self, strength=5):
        bqm_hubo = self.full_hubo.copy()
        bqm_hubo.normalize()
        problem_dict, off = bqm_hubo.to_hubo()
        self.bqm = dimod.make_quadratic(problem_dict, strength = strength, vartype = dimod.Vartype.BINARY)
    
    
    def solve_with_exact_poly_solver(self):
        solver = dimod.ExactPolySolver()
        result = solver.sample_poly(self.full_hubo)
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
        result = sampler.sample(self.bqm, num_reads=1000)
        self.samplesets["simulated_annealing"] = result
        return result
    
    
    def solve_with_steepest_descent(self):
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
                model.Params.LogFile = "logs//" + self.name +  "_gurobi.log"
                model.Params.OutputFlag = 1
                model.Params.LogLevel = 1
                model.Params.MIPFocus = 0 # aims to find a single optimal solution
                model.Params.PoolSearchMode = 0 # No need for multiple solutions
                model.Params.PoolGap = 0.0 # Only provably optimal solutions are added to the pool
                model.Params.TimeLimit = 240
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
    
    
    def solve_with_qaoa_pennylane(self):
        qaoa = QuantumApproximateOptimizationAlgorithm(self.bqm, {})
        qaoa.solve_with_QAOA_pennylane()
        return qaoa.samplesets["qaoa_pennylane"]
    
    
    def get_number_of_hubo_variables(self):
        return len(self.full_hubo.variables)
    
    def get_number_of_bqm_variables(self):
        return len(self.bqm.variables)
    
    def get_number_of_hubo_terms(self):
        return len(self.full_hubo)
    
    def get_number_of_bqm_terms(self):
        return len(self.bqm.linear) + len(self.bqm.quadratic)
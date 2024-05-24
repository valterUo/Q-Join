import itertools
import time

import numpy as np

from classical_algorithms.weights_costs import basic_cost
from utils import append_to_json, build_nested_list, compare_nested_lists, store_gurobi_results


class Solver:
    
    def __init__(self, qjoin, experiment_name) -> None:
        self.qjoin = qjoin
        self.experiment_name = experiment_name
        self.query_graph = qjoin.query_graph
        
    
    def solve(self, solver):
        if solver == "compute_variable_statistics":
            self.compute_variable_statistics()
        
        if solver == "exact_poly_solver":
            self.solve_with_exact_poly_solver()
        
        if solver == "exact_bqm_solver":
            self.solve_with_exact_BQM_solver()
        
        if solver == "tabu_sampler":
            self.solve_with_TabuSampler()
        
        if solver == "simulated_annealing":
            self.solve_with_simulated_annealing()
        
        if solver == "gurobi":
            self.solve_with_Gurobi()
        
        if solver == "dynamic_programming":
            self.solve_with_dynamic_programming()
            
        if solver == "qaoa_pennylane":
            self.solve_with_QAOA_pennylane()
    
    
    def compute_variable_statistics(self):
        result = {}
        result["hubo_variables"] = self.qjoin.get_number_of_hubo_variables()
        result["hubo_terms"] = self.qjoin.get_number_of_hubo_terms()
        result["bqm_variables"] = self.qjoin.get_number_of_bqm_variables()
        result["bqm_terms"] = self.qjoin.get_number_of_bqm_terms()
        result["number_of_nodes"] = len(self.query_graph.nodes)
        result["number_of_edges"] = len(self.query_graph.edges)
        avarage_join_tree_cost = 0
        permutations = itertools.permutations(list(range(len(self.query_graph.nodes))))
        for i in range(1000):
            try:
                perm = next(permutations)
            except StopIteration:
                break
            join = [perm[0], perm[1]]
            for i in range(2, len(perm)):
                join = [join, perm[i]]
            avarage_join_tree_cost += basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        avarage_join_tree_cost /= i
        result["avarage_join_tree_cost"] = avarage_join_tree_cost
        append_to_json(self.experiment_name, str(self.query_graph), result)
    
    
    def solve_with_exact_poly_solver(self):
        start_time = time.time()
        quantum_result = self.qjoin.solve_with_exact_poly_solver()
        end_time = time.time()
       
        poly_res = {}
        for res in quantum_result.first.sample:
            if quantum_result.first.sample[res] == 1 and "*" not in res:
                poly_res[res] = 1
        
        tuples = list(poly_res.keys())
        join = build_nested_list(tuples)
        poly_res = {str(k) : v for k, v in poly_res.items()}
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        classic_solution = self.qjoin.solve_with_dynamic_programming()
        
        stored_result = {"solution": poly_res, 
                         "join" : join, 
                         "cost": classical_cost, 
                         "energy": quantum_result.first.energy, 
                         "time": end_time - start_time, 
                         "optimal_cost": classic_solution[1], 
                         "optimal_solution": classic_solution[0],
                         "found_optimal": bool(np.isclose(classical_cost, classic_solution[1], atol=1e-5)),
                         "plans_are_equal": compare_nested_lists(join, classic_solution[0])}
        
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        
        
    def solve_with_Gurobi(self):
        start_time = time.time()
        quantum_result = self.qjoin.solve_with_Gurobi()
        end_time = time.time()
        
        gurobi_res = {}
        for var in quantum_result["result"]:
            if quantum_result["result"][var] == 1 and "a" not in var:
                var = "".join(var.split("_"))
                gurobi_res[eval(var)] = 1
        
        gurobi_res_pos = gurobi_res.copy()
        gurobi_res_pos = {str(k): v for k, v in gurobi_res_pos.items() if v == 1}
        
        tuples = list(gurobi_res.keys())
        join = build_nested_list(tuples)
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        for v in self.qjoin.full_hubo.variables:
            if v not in gurobi_res:
                gurobi_res[v] = 0
            
        quantum_cost = self.qjoin.evaluate_cost(gurobi_res)
        classic_solution = self.qjoin.solve_with_dynamic_programming()
        
        stored_result = {"solution": gurobi_res_pos, 
                         "join" : join, 
                         "cost": classical_cost, 
                         "time": end_time - start_time, 
                         "optimal_cost": classic_solution[1], 
                         "optimal_solution": classic_solution[0],
                         "found_optimal": bool(np.isclose(quantum_cost, classic_solution[1], atol=1e-5)),
                         "plans_are_equal": compare_nested_lists(join, classic_solution[0])}
        
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
    
    
    def solve_with_exact_BQM_solver(self):
        start_time = time.time()
        quantum_result = self.qjoin.solve_with_exact_BQM_solver()
        end_time = time.time()
        bqm_res = {}
        for res in quantum_result.first.sample:
            if quantum_result.first.sample[res] == 1 and "*" not in res:
                bqm_res[res] = 1
        tuples = list(bqm_res.keys())
        join = build_nested_list(tuples)
        bqm_res = {str(k) : v for k, v in bqm_res.items()}
        stored_result = {"solution": bqm_res, "join": join, "cost": quantum_result.first.energy, "time": end_time - start_time}
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
    
    
    def solve_with_TabuSampler(self):
        start_time = time.time()
        quantum_result = self.qjoin.solve_with_TabuSampler(1000)
        end_time = time.time()
        tabu_res = {}
        for var in quantum_result.first.sample:
            if quantum_result.first.sample[var] == 1 and "*" not in var:
                tabu_res[var] = 1
        tuples = list(tabu_res.keys())
        join = build_nested_list(tuples)
        tabu_res = {str(k) : v for k, v in tabu_res.items()}
        stored_result = {"solution": tabu_res, "join": join, "cost": quantum_result.first.energy, "time": end_time - start_time}
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
    
    
    def solve_with_simulated_annealing(self):
        start_time = time.time()
        quantum_result = self.qjoin.solve_with_simulated_annealing()
        end_time = time.time()
        sim_res = {}
        for var in quantum_result.first.sample:
            if quantum_result.first.sample[var] == 1 and "*" not in var:
                sim_res[var] = 1
        tuples = list(sim_res.keys())
        join = build_nested_list(tuples)
        sim_res = {str(k) : v for k, v in sim_res.items()}
        stored_result = {"solution": sim_res, "join" : join, "cost": quantum_result.first.energy, "time": end_time - start_time}
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
    
    
    def solve_with_dynamic_programming(self):
        dynamic_start_time = time.time()
        classic_solution = self.qjoin.solve_with_dynamic_programming()
        dynamic_end_time = time.time()
        stored_result = {"solution": classic_solution[0], "cost": classic_solution[1], "time": dynamic_end_time - dynamic_start_time}
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        
        
    def solve_with_QAOA_pennylane(self):
        quantum_result = self.qjoin.solve_with_QAOA_pennylane()
        append_to_json(self.experiment_name, str(self.query_graph), quantum_result)
        
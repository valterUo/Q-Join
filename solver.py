import itertools
import json
import time

import numpy as np

from classical_algorithms.weights_costs import basic_cost
from utils import append_to_json, build_nested_list, compare_nested_lists, flatten, store_gurobi_results


class Solver:
    
    def __init__(self, qjoin, experiment_name, method_name) -> None:
        self.qjoin = qjoin
        self.experiment_name = experiment_name
        self.query_graph = qjoin.query_graph
        self.method_name = method_name
        
    
    def solve(self, solver):
        if solver == "compute_variable_statistics":
            self.compute_variable_statistics()
        
        if solver == "exact_poly_solver":
            return self.solve_with_exact_poly_solver()
        
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
            
        if solver == "qaoa_qiskit":
            self.solve_with_QAOA_qiskit()
            
        if solver == "dwave_LeapHybridSampler":
            self.solve_with_dwave("LeapHybridSampler")
        
        if solver == "dwave_DWaveSampler":
            self.solve_with_dwave("DWaveSampler")
        
        if solver == "dwave_Kerberos":
            self.solve_with_dwave("Kerberos")
    
    
    def compute_variable_statistics(self):
        result = {}
        result["hubo_variables"] = self.qjoin.get_number_of_hubo_variables()
        result["hubo_terms"] = self.qjoin.get_number_of_hubo_terms()
        result["bqm_variables"] = self.qjoin.get_number_of_bqm_variables()
        result["bqm_terms"] = self.qjoin.get_number_of_bqm_terms()
        result["number_of_nodes"] = len(self.query_graph.nodes)
        result["number_of_edges"] = len(self.query_graph.edges)
        result["method"] = self.qjoin.method_name
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
            if quantum_result.first.sample[res] == 1:
                #print(res)
                if "*" not in res and "a" not in res and len(res) > 2:
                    poly_res[res] = 1
                    
        # print all results with minimum energy
        #lowest_energy_samples = quantum_result.lowest()
        
        #for sample in lowest_energy_samples:
        #    pos_vars = [v for v in sample if sample[v] == 1]
            # Sort with respect to the last component
        #    pos_vars = sorted(pos_vars, key=lambda x: x[-1])
        #    for var in pos_vars:
        #        print(var)
        #    print("\n")
        
        tuples = list(poly_res.keys())
        
        if self.method_name == "presice_2":
            # Given tuples (0, 1, 0), (0, 1, 1), (1, 2, 1), take only tuples that are new for each third component
            group_by_third = {}
            for t in tuples:
                if t[2] not in group_by_third:
                    group_by_third[t[2]] = [t]
                else:
                    group_by_third[t[2]].append(t)
            new_tuples = []
            for k in group_by_third:
                if k == 0:
                    new_tuples.append(group_by_third[k][0])
                else:
                    grouped_tuples = group_by_third[k]
                    for t in grouped_tuples:
                        if t not in new_tuples:
                            new_tuples.append(t)
                            break
            tuples = new_tuples
            #print(tuples)
        
        print("result", tuples)
        join = build_nested_list(tuples)
        poly_res = [{str(k) : v for k, v in poly_res.items()}]
        print(json.dumps(join, indent=4))
        
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        
        estimation_size = self.qjoin.get_estimation_size()
        dynamic_programming_solution = self.qjoin.solve_with_dynamic_programming()
        graph_aware_dynamic_programming = self.qjoin.solve_with_graph_aware_dynamic_programming()
        greedy_solution = self.qjoin.solve_with_greedy()
        graph_aware_greedy_solution = self.qjoin.solve_with_greedy_with_query_graph()
        
        
        stored_result = {"solution": poly_res, 
                         "join" : join, 
                         "cost": classical_cost, 
                         "energy": quantum_result.first.energy,
                         "estimation_size": estimation_size,
                         "time": end_time - start_time, 
                         "dynamic_programming_cost": dynamic_programming_solution[1], 
                         "dynamic_programming_solution": dynamic_programming_solution[0],
                         "graph_aware_dynamic_programming_cost": graph_aware_dynamic_programming[1],
                         "graph_aware_dynamic_programming_solution": graph_aware_dynamic_programming[0],
                         "greedy_cost": greedy_solution[1],
                         "greedy_solution": greedy_solution[0],
                         "graph_aware_greedy_cost": graph_aware_greedy_solution[1],
                         "graph_aware_greedy_solution": graph_aware_greedy_solution[0],
                         "found_optimal": bool(np.isclose(classical_cost, dynamic_programming_solution[1], atol=1e-5)),
                         "plans_are_equal": compare_nested_lists(join, dynamic_programming_solution[0]),}
        
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        return quantum_result.first.sample
        
        
    def solve_with_Gurobi(self):
        start_time = time.time()
        quantum_result = self.qjoin.solve_with_Gurobi()
        end_time = time.time()
        
        gurobi_res = {}
        for var in quantum_result["result"]:
            if quantum_result["result"][var] == 1:
                #print(var)
                if "a" not in var:
                    var = "".join(var.split("_"))
                    gurobi_res[eval(var)] = 1
        
        gurobi_res_pos = gurobi_res.copy()
        gurobi_res_pos = {k: v for k, v in gurobi_res_pos.items() if v == 1}
        gurobi_res_pos = sorted(gurobi_res_pos, key=lambda x: (x[0], x[1], x[2]))
        tuples = gurobi_res_pos #list(gurobi_res.keys())
        
        if self.method_name == "precise_2" and self.qjoin.query_graph_name != "clique":
            #print(tuples)
            # Given tuples (0, 1, 0), (0, 1, 1), (1, 2, 1), take only tuples that are new for each third component
            group_by_third = {}
            for t in tuples:
                if t[2] not in group_by_third:
                    group_by_third[t[2]] = [t]
                else:
                    group_by_third[t[2]].append(t)
            new_tuples = []
            for k in range(len(group_by_third)):
                if k == 0:
                    new_tuples.append(group_by_third[k][0])
                else:
                    grouped_tuples = group_by_third[k]
                    for t in grouped_tuples:
                        if (t[0], t[1]) not in [(x[0], x[1]) for x in new_tuples]:
                            if t[0] in [x[0] for x in new_tuples]:
                                new_tuples.append(t)
                                break 
                            elif t[1] in [x[1] for x in new_tuples]:
                                new_tuples.append(t)
                                break
                            elif t[0] in [x[1] for x in new_tuples]:
                                new_tuples.append(t)
                                break
                            elif t[1] in [x[0] for x in new_tuples]:
                                new_tuples.append(t)
                                break   
            tuples = new_tuples
            print(group_by_third)
            print(tuples)
        
        #tuples = list(gurobi_res.keys())
        join = build_nested_list(tuples)
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        for v in self.qjoin.full_hubo.variables:
            if v not in gurobi_res:
                gurobi_res[v] = 0
            
        quantum_cost = self.qjoin.evaluate_cost(gurobi_res)
        
        found_optimal = 0
        if len(self.query_graph.nodes) < 17 or self.qjoin.query_graph_name != "clique":
            classic_solution = self.qjoin.solve_with_dynamic_programming()
            found_optimal = bool(np.isclose(quantum_cost, classic_solution[1], atol=1e-5))
        else:
            classic_solution = [0, 0]
            #graph_aware_dynamic_programming = [0, 0]
            
        greedy_solution_with_graph = self.qjoin.solve_with_greedy_with_query_graph()
        estimation_size = self.qjoin.get_estimation_size()
        
        graph_aware_dynamic_programming = self.qjoin.solve_with_graph_aware_dynamic_programming()
        
        if len(tuples) != len(self.query_graph.nodes) - 1:
            print("ERROR")
            print(tuples)
            return None
        
        tables_in_result = set([t[0] for t in tuples]).union(set([t[1] for t in tuples]))
        if len(tables_in_result) != len(self.query_graph.nodes):
            print("ERROR")
            print(tables_in_result)
            return None
        
        greedy_solution = self.qjoin.solve_with_greedy()
        
        print(json.dumps(join, indent=4))
        gurobi_res_pos = [{str(k) : 1 for k in gurobi_res_pos}]
        stored_result = {"solution": gurobi_res_pos, 
                         "join" : join, 
                         "cost": classical_cost, 
                         "time": end_time - start_time,
                         "estimation_size": estimation_size,
                         "optimal_cost": classic_solution[1], 
                         "optimal_solution": classic_solution[0],
                         "graph_aware_dynamic_programming_cost": graph_aware_dynamic_programming[1],
                         "graph_aware_dynamic_programming_solution": graph_aware_dynamic_programming[0],
                         "greedy_cost": greedy_solution[1],
                         "greedy_solution": greedy_solution[0],
                         "graph_aware_greedy_cost": greedy_solution_with_graph[1],
                         "graph_aware_greedy_solution": greedy_solution_with_graph[0],
                         "found_optimal": found_optimal,
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
        print(quantum_result.first.energy)
        for var in quantum_result.first.sample:
            if quantum_result.first.sample[var] == 1: #and "*" not in var:
                print(var)
                tabu_res[var] = 1
        #tuples = list(tabu_res.keys())
        #join = build_nested_list(tuples)
        #tabu_res = {str(k) : v for k, v in tabu_res.items()}
        #stored_result = {"solution": tabu_res, "join": join, "cost": quantum_result.first.energy, "time": end_time - start_time}
        #append_to_json(self.experiment_name, str(self.query_graph), stored_result)
    
    
    def solve_with_simulated_annealing(self):
        start_time = time.time()
        quantum_result = self.qjoin.solve_with_simulated_annealing()
        end_time = time.time()
        sim_res = {}
        print(quantum_result.first.energy)
        for var in quantum_result.first.sample:
            if quantum_result.first.sample[var] == 1 and "*" not in var and "a" not in var:
                sim_res[var] = 1
        
        tuples = list(sim_res.keys())
        # sort the tuples so that vars containing "a" are at the end
        tuples = sorted(tuples, key=lambda x: "a" in x)
        #for t in tuples:
        #    print(t)
        join = build_nested_list(tuples)
        
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        for v in self.qjoin.full_hubo.variables:
            if v not in sim_res:
                sim_res[v] = 0
            
        quantum_cost = self.qjoin.evaluate_cost(sim_res)
        
        found_optimal = 0
        if len(self.query_graph.nodes) < 17:
            classic_solution = self.qjoin.solve_with_dynamic_programming()
            found_optimal = bool(np.isclose(quantum_cost, classic_solution[1], atol=1e-5))
        else:
            classic_solution = [0, 0]
            
        greedy_solution_with_graph = self.qjoin.solve_with_greedy_with_query_graph()
        estimation_size = self.qjoin.get_estimation_size()
        
        graph_aware_dynamic_programming = self.qjoin.solve_with_graph_aware_dynamic_programming()
        
        #if not np.isclose(classical_cost, graph_aware_dynamic_programming[1]):
        #    print("ERROR")
        #    return None
        
        greedy_solution = self.qjoin.solve_with_greedy()
        sim_res = [tuple(k) for k, v in sim_res.items() if v == 1]
        sim_res = sorted(sim_res, key=lambda x : x[-1])
        stored_result = {"solution": sim_res, 
                         "join" : join, 
                         "cost": classical_cost, 
                         "time": end_time - start_time,
                         "estimation_size": estimation_size,
                         "optimal_cost": classic_solution[1], 
                         "optimal_solution": classic_solution[0],
                         "graph_aware_dynamic_programming_cost": graph_aware_dynamic_programming[1],
                         "graph_aware_dynamic_programming_solution": graph_aware_dynamic_programming[0],
                         "greedy_cost": greedy_solution[1],
                         "greedy_solution": greedy_solution[0],
                         "graph_aware_greedy_cost": greedy_solution_with_graph[1],
                         "graph_aware_greedy_solution": greedy_solution_with_graph[0],
                         "found_optimal": found_optimal,
                         "plans_are_equal": compare_nested_lists(join, classic_solution[0])}

        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        return quantum_result.first.sample
    
    
    def solve_with_dynamic_programming(self):
        dynamic_start_time = time.time()
        classic_solution = self.qjoin.solve_with_dynamic_programming()
        dynamic_end_time = time.time()
        stored_result = {"solution": classic_solution[0], "cost": classic_solution[1], "time": dynamic_end_time - dynamic_start_time}
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        
        
    def solve_with_QAOA_pennylane(self):
        quantum_result = self.qjoin.solve_with_qaoa_pennylane()
        
        sim_res = {}
        
        for var in quantum_result["result"]:
            if quantum_result["result"][var] == 1 and "*" not in var:
                sim_res[eval(var)] = 1
        
        tuples = list(sim_res.keys())
        join = build_nested_list(tuples)
        sim_res = {str(k) : v for k, v in sim_res.items()}
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        classic_solution = self.qjoin.solve_with_dynamic_programming()
        
        stored_result = {"join" : join, 
                         "qaoa": quantum_result, 
                         "cost": classical_cost, 
                         "optimal_cost": classic_solution[1], 
                         "optimal_solution": classic_solution[0],
                         "found_optimal": bool(np.isclose(classical_cost, classic_solution[1], atol=1e-5)),
                         "plans_are_equal": compare_nested_lists(join, classic_solution[0])}
        
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        
    
    def solve_with_QAOA_qiskit(self):
        quantum_result = self.qjoin.solve_with_qaoa_qiskit()
        
        print(quantum_result)
        qaoa_cost = float(quantum_result["optimal_cost"].real)
        probability = quantum_result["probability"]
        #initial_point = quantum_result["initial_point"]
        sim_res = {}
        for var in quantum_result["result"]:
            if quantum_result["result"][var] == 1 and "*" not in var:
                sim_res[var] = 1
        tuples = list(sim_res.keys())
        join = build_nested_list(tuples)
        sim_res = {str(k) : v for k, v in sim_res.items()}
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        classic_solution = self.qjoin.solve_with_dynamic_programming()
        stored_result = {"join" : join,
                         "qaoa": {str(k) : v for k, v in quantum_result["result"].items()},
                         "cost": classical_cost,
                         "qaoa_cost" : qaoa_cost,
                         "probability": probability,
                         "optimal_cost": classic_solution[1], 
                         "optimal_solution": classic_solution[0],
                         "found_optimal": bool(np.isclose(classical_cost, classic_solution[1], atol=1e-5)),
                         "plans_are_equal": compare_nested_lists(join, classic_solution[0]) }
        
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        
        
    def solve_with_dwave(self, sampler):
        
        if sampler == "LeapHybridSampler":
            quantum_result = self.qjoin.solve_with_LeapHybridSampler()
        elif sampler == "DWaveSampler":
            quantum_result = self.qjoin.solve_with_DWave_Sampler()
        elif sampler == "Kerberos":
            quantum_result = self.qjoin.solve_with_KerberosSampler()
            
        res = {}
        for var in quantum_result.first.sample:
            if quantum_result.first.sample[var] == 1 and "*" not in var and "a" not in var:
                res[var] = 1
        tuples = list(res.keys())   
        if self.method_name == "precise_2" and self.qjoin.query_graph_name != "clique":
            #print(tuples)
            # Given tuples (0, 1, 0), (0, 1, 1), (1, 2, 1), take only tuples that are new for each third component
            group_by_third = {}
            for t in tuples:
                if t[2] not in group_by_third:
                    group_by_third[t[2]] = [t]
                else:
                    group_by_third[t[2]].append(t)
            new_tuples = []
            for k in range(len(group_by_third)):
                if k == 0:
                    new_tuples.append(group_by_third[k][0])
                else:
                    grouped_tuples = group_by_third[k]
                    for t in grouped_tuples:
                        if (t[0], t[1]) not in [(x[0], x[1]) for x in new_tuples]:
                            if t[0] in [x[0] for x in new_tuples]:
                                new_tuples.append(t)
                                break 
                            elif t[1] in [x[1] for x in new_tuples]:
                                new_tuples.append(t)
                                break
                            elif t[0] in [x[1] for x in new_tuples]:
                                new_tuples.append(t)
                                break
                            elif t[1] in [x[0] for x in new_tuples]:
                                new_tuples.append(t)
                                break   
            tuples = new_tuples
            #print(group_by_third)
            #print(tuples)
        
        join = build_nested_list(tuples)
        classical_cost = basic_cost(join, self.qjoin.relations, self.qjoin.selectivities)
        quantum_cost = 0
        res = [{str(k) : v for k, v in res.items()}]
        
        estimation_size = self.qjoin.get_estimation_size()
        dynamic_programming_solution = self.qjoin.solve_with_dynamic_programming()
        graph_aware_dynamic_programming = self.qjoin.solve_with_graph_aware_dynamic_programming()
        greedy_solution = self.qjoin.solve_with_greedy()
        graph_aware_greedy_solution = self.qjoin.solve_with_greedy_with_query_graph()
        
        flattened_join = list(flatten(join))
        greedy_sol = list(flatten(greedy_solution[0]))
        
        if len(tuples) != len(self.query_graph.nodes) - 1:
            print("ERROR")
            return None
        
        tables_in_result = set([t[0] for t in tuples]).union(set([t[1] for t in tuples]))
        if len(tables_in_result) != len(self.query_graph.nodes):
            print("ERROR")
            return None
        
        stored_result = {"solution": res, 
                         "join" : join, 
                         "cost": classical_cost,
                         "energy": quantum_result.first.energy,
                         "estimation_size": estimation_size,
                         "dynamic_programming_cost": dynamic_programming_solution[1], 
                         "dynamic_programming_solution": dynamic_programming_solution[0],
                         "graph_aware_dynamic_programming_cost": graph_aware_dynamic_programming[1],
                         "graph_aware_dynamic_programming_solution": graph_aware_dynamic_programming[0],
                         "greedy_cost": greedy_solution[1],
                         "greedy_solution": greedy_solution[0],
                         "graph_aware_greedy_cost": graph_aware_greedy_solution[1],
                         "graph_aware_greedy_solution": graph_aware_greedy_solution[0],
                         "found_optimal": bool(np.isclose(quantum_cost, dynamic_programming_solution[1], atol=1e-5)),
                         "plans_are_equal": compare_nested_lists(join, dynamic_programming_solution[0])}
        
        append_to_json(self.experiment_name, str(self.query_graph), stored_result)
        
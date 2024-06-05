import time
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from qiskit.circuit.library import QAOAAnsatz
np.random.seed(0)

import sys
sys.path.append('qaoa/')  # Replace '/path/to/qaoa_pennylane' with the actual path to the qaoa_pennylane module

from qiskit import primitives
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, ADAM, GradientDescent, AQGD
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_distribution

from qaoa_pennylane import QAOAPennylane
from qaoa_qiskit import QAOAQiskit

from scipy.optimize import minimize

import jax
import jax.numpy as jnp
import optax
from jax import random

class QuantumApproximateOptimizationAlgorithm:
    
    def __init__(self, bqm, samplesets) -> None:
        self.bqm = bqm
        self.samplesets = samplesets
        
        
    def solve_with_QAOA_pennylane(self):
        
        def add_parameters(parameters, k):
            new_params = np.random.uniform(-2 * np.pi, 2 * np.pi, k)
            #print("New params: ", new_params)
            #print("Old params: ", parameters)
            # Append new params right before the last parameter
            parameters = np.insert(parameters, -1, new_params, axis=0)
            #print("Parameters after update: ", parameters)
            return parameters
        
        start_time = time.time()
        depth = 1
        H = QAOAPennylane(self.bqm, depth)
        circuit = H.get_circuit()
        offset = H.get_offset()
        num_of_qubits = len(H.get_variables())
        #print("Number of qubits: ", num_of_qubits)
        number_of_terms = H.get_number_of_terms()
        #print("Number of terms: ", number_of_terms)
        #optimizer = optax.adam() #qml.SPSAOptimizer() #qml.NesterovMomentumOptimizer() #qml.GradientDescentOptimizer()
        steps = 40
        
        key = random.PRNGKey(99)
        #max_degree = max([len(term) for term in H.get_variables()])
        mixer_value = np.random.uniform(-2 * np.pi, 2 * np.pi, 1)
        params = [mixer_value]
        step = 20
        for k in range(step, number_of_terms, step):
            print("Number of included terms: ", k)
            H = QAOAPennylane(self.bqm, depth, n_included_terms = k)
            circuit = H.get_circuit()
            
            #params = random.uniform(key=key, 
            #                        minval = -2*np.pi, 
            #                        maxval = 2*np.pi, 
            #                        shape = (depth, number_of_terms + 1))
            
            new_params = []
            for i in range(depth):
                new_params.append(add_parameters(params[i], step))
            params = np.array(new_params) #, requires_grad=True)
            
            #fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(params)
            #fig.savefig(f"QAOA_circuit_pennylane_{k}.png")
            
            #print("Parameters: ", params)
            #optimizer = qml.GradientDescentOptimizer(stepsize=0.01)
            
            #optimizer = optax.polyak_sgd(max_learning_rate=0.1, f_min = -265.17)
            #optimizer = optax.adamw(learning_rate=0.01)
            #optimizer = optax.rmsprop(learning_rate=0.9) #optax.adamaxw(learning_rate=0.05) #, b1 = 0.7, b2 = 0.9)
            #opt_state = optimizer.init(params)
            
            #for i in range(steps):
            #    params, current_cost = optimizer.step_and_cost(circuit, params)
            #    if i % 100 == 0:
            #        print("Step: ", i, "Cost: ", current_cost)
            
            #print(circuit(params))
            
            def update_step(opt, params, opt_state):
                loss_val, grads = jax.value_and_grad(circuit)(params)
                #print("Loss: ", loss_val)
                updates, opt_state = opt.update(grads, opt_state, params)
                #print("Updates: ", updates)
                params = optax.apply_updates(params, updates)
                #print("Params: ", params)
                return params, opt_state, loss_val
            
            #optimizer = qml.SPSAOptimizer(maxiter=100, alpha=0.000602, gamma=0.000101, c = 0.001)
            #for i in range(steps):
                #print("Step: ", i)
                #params, opt_state, loss_val = update_step(optimizer, params, opt_state)
                #if i % 1 == 0:
                #    print(f"Step: {i} Loss: {loss_val}")
                    #params, current_cost = optimizer.step_and_cost(circuit, params)
                    #if i % 1 == 0:
                    #    print("Step: ", i, "Cost: ", current_cost)
                    
            #options={'maxiter': 100}     
            result = minimize(circuit, params[0], method="COBYLA", callback=lambda x: print("Cost: ", circuit(x))) #, options=options)
            #print("Result: ", result.x)
            params = [result.x]
        
        get_probability_circuit = H.get_probability_circuit()
        probs = get_probability_circuit(params)
        max_arg = np.argmax(probs)
        result_bin = np.binary_repr(max_arg, width=num_of_qubits)
        end_time = time.time()
        solution_prob = float(probs[max_arg])
        print("Solution probability: ", solution_prob)
        first_excited_state_prob = float(probs[np.argsort(probs)[-2]])
        print("First excited state probability: ", first_excited_state_prob)
        qubits_to_variables = H.get_qubits_to_variables()
        
        result = {}
        for i, b in enumerate(result_bin):
            result[qubits_to_variables[i]] = (int(b) + 1) % 2
        
        #result = dict(zip(H.get_variables(), [int(i) for i in result_bin]))
        result = {str(k) : v for k, v in result.items()}
        
        self.samplesets["qaoa_pennylane"] = { "result": result, 
                                             "n_qubits": num_of_qubits, 
                                             "solution_prob": solution_prob,
                                             "first_excited_state_prob": first_excited_state_prob,
                                             "offset": offset,
                                             "time": end_time - start_time }
        #for var, value in self.samplesets["qaoa_pennylane"].items():
        #    print(var, type(value))
        return self.samplesets["qaoa_pennylane"]
        
    
    def solve_with_QAOA_Qiskit1(self):
        qaoa_qiskit = QAOAQiskit(self.bqm)
        H = qaoa_qiskit.get_Hamiltonian()
        
        sampler = primitives.Sampler()
        
        qubits_to_variables = qaoa_qiskit.get_qubits_to_variables()
        qaoa = QAOA(sampler, COBYLA(maxiter=1000))
        print("Start QAOA computation...")
        result = qaoa.compute_minimum_eigenvalue(H)
        print("Result: ", result)

        if False:
            ansatz = qaoa.ansatz
            ansatz.decompose(reps=3).draw(output="mpl", style="iqp", filename="QAOA_circuit_qiskit.png")
        
        best_bitstring = result.best_measurement["bitstring"]
        optimal_cost = result.best_measurement['value']
        probability = result.best_measurement['probability']
        initial_point = str(qaoa.initial_point)
        
        # Because the problem was encoded with Ising model, 
        # we convert the result back to QUBO: |0> -> 1 and |1> -> -1
        # This means that the measured 0s represent the solution to the problem 
        #solution = dict(zip(reversed(labels), [(int(i) + 1) % 2 for i in best_bitstring]))
        
        result = {}
        for i, b in enumerate(best_bitstring):
            result[qubits_to_variables[i]] = (int(b) + 1) % 2
        
        self.samplesets["qaoa_qiskit"] = { "result": result, 
                                          "offset": qaoa_qiskit.get_offset(),
                                          "optimal_cost": optimal_cost,
                                          "probability": probability,
                                          "initial_point": initial_point }
        return result
    
    
    def solve_with_QAOA_Qiskit2(self):
        qaoa_qiskit = QAOAQiskit(self.bqm)
        H = qaoa_qiskit.get_Hamiltonian()
        
        if True:
            # Convert SparsePauliOp to a dense matrix
            matrix = H.to_matrix()

            # Compute eigenvalues and eigenvectors using NumPy
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)

            # Find the minimum eigenvalue and its corresponding eigenvector
            min_index = np.argmin(eigenvalues)
            min_eigenvalue = eigenvalues[min_index]
            min_eigenvector = eigenvectors[:, min_index]

            # Print the minimum eigenvalue and its corresponding eigenvector
            print("Minimum Eigenvalue:")
            print(min_eigenvalue)
            print("Corresponding Eigenvector:")
            print(min_eigenvector)
            index_1 = np.min(np.nonzero(min_eigenvector == 1)[0])
            bitstring = np.binary_repr(index_1, width=len(qaoa_qiskit.get_variables()))
            print("Corresponding bitstring:")
            print(bitstring)
            res = {qaoa_qiskit.get_qubits_to_variables()[i]: (int(b) + 1) % 2 for i, b in enumerate(bitstring)}
            print("Analytic solution:")
            for r in res:
                print(r, res[r])
        
        depth = 1
        mixer_operator = [primitives.PauliX(i) for i in range(len(qaoa_qiskit.get_variables()))]
        ansatz = QAOAAnsatz(H, reps=depth)
        
        if True:
            ansatz.decompose(reps=3).draw(output="mpl", style="iqp", filename="QAOA_circuit_qiskit.png")
        
        backend = AerSimulator(method='statevector', device = "GPU")
        
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=0)
        ansatz_isa = pm.run(ansatz)
        hamiltonian_isa = H.apply_layout(ansatz_isa.layout)
        qubits_to_variables = qaoa_qiskit.get_qubits_to_variables()
        estimator = Estimator(backend=backend)
        sampler = Sampler(backend=backend)
        
        def cost_func(params):
            pub = (ansatz_isa, [hamiltonian_isa], [params])
            result = estimator.run(pubs=[pub]).result()
            cost = result[0].data.evs[0]
            #print("Cost: ", cost)
            return cost
            
        x0 = np.random.uniform(-2*np.pi, 2*np.pi, 2*depth)
        #optimizer = ADAM() #(maxiter=1000, learning_rate = 0.00001, callback = lambda x, y, z, k: print("Cost: ", z))
        result = minimize(cost_func, x0, method="CG", callback=lambda x: print("Cost: ", cost_func(x)), tol=1e-9)
        #result = optimizer.minimize(fun=cost_func, x0=x0)
        print("Result: ", result)
        
        # Evaluate the optimal parameters and obtain probs
        # Assign solution parameters to ansatz
        qc = ansatz.assign_parameters(result.x)
        # Add measurements to our circuit
        qc.measure_all()
        qc_isa = pm.run(qc)
        sresult = sampler.run([qc_isa]).result()
        samp_dist = sresult[0].data.meas.get_counts()
        plot_distribution(samp_dist, figsize=(15, 5), title="QAOA Qiskit", filename="QAOA_qiskit_distribution.png")
        
        # QAOA ansatz circuit
        if False:
            ansatz = qaoa.ansatz
            ansatz.decompose(reps=3).draw(output="mpl", style="iqp", filename="QAOA_circuit_qiskit.png")
        
        # Get the best bitstring with most counts in samp_dist
        best_bitstring = max(samp_dist, key=samp_dist.get)
        optimal_cost = result.fun
        probability = samp_dist[best_bitstring]/sum(samp_dist.values())
        
        result = {}
        for i, b in enumerate(best_bitstring):
            result[qubits_to_variables[i]] = (int(b) + 1) % 2
        
        self.samplesets["qaoa_qiskit"] = { "result": result, 
                                          "offset": qaoa_qiskit.get_offset(),
                                          "optimal_cost": optimal_cost,
                                          "probability": probability }
        return result
    
    def get_samplesets(self):
        return self.samplesets
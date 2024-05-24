import time
import numpy as np
import pennylane as qml
from pennylane import numpy as nnp
nnp.random.seed(0)

import sys
sys.path.append('qaoa/')  # Replace '/path/to/qaoa_pennylane' with the actual path to the qaoa_pennylane module

from qiskit import primitives
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA, ADAM, GradientDescent, AQGD

from qaoa_pennylane import QAOAPennylane
from qaoa_qiskit import QAOAQiskit

class QuantumApproximateOptimizationAlgorithm:
    
    def __init__(self, bqm, samplesets) -> None:
        self.bqm = bqm
        self.samplesets = samplesets
        
        
    def solve_with_QAOA_pennylane(self):
        start_time = time.time()
        depth = 2
        H = QAOAPennylane(self.bqm, depth)
        circuit = H.get_circuit()
        offset = H.get_offset()
        num_of_qubits = len(H.get_variables())
        optimizer = qml.GradientDescentOptimizer() #qml.NesterovMomentumOptimizer()
        steps = 1000
        params = 0.1 * nnp.random.rand(2, depth, requires_grad=True) #nnp.array([[0.5]*depth, [0.5]*depth], requires_grad=True)

        fig, ax = qml.draw_mpl(circuit)(params)
        fig.savefig("QAOA_circuit_pennylane.png")
        
        for _ in range(steps):
            params = optimizer.step(circuit, params)
        # Get the final value
        energy = circuit(params)
        get_probability_circuit = H.get_probability_circuit()
        probs = get_probability_circuit(params)

        #plt.bar(range(2 ** num_of_qubits), probs[1:])
        #plt.show()
        
        max_arg = np.argmax(probs)
        result_bin = np.binary_repr(max_arg, width=num_of_qubits)
        end_time = time.time()
        # get the probability of the solution
        solution_prob = probs[max_arg]
        #print("Solution probability: ", solution_prob)
        # get the probability of the second most probable solution
        first_excited_state_prob = probs[np.argsort(probs)[-2]]
        #print("First excited state probability: ", first_excited_state_prob)
        # Because the problem was encoded with Ising model, 
        # we convert the result back to QUBO: |0> -> 1 and |1> -> -1
        # This means that the measured 0s represent the solution to the problem
        result = dict(zip(H.get_variables(), [(int(i) + 1) % 2 for i in result_bin]))
        self.samplesets["qaoa_pennylane"] = { "result": result, 
                                             "n_qubits": num_of_qubits, 
                                             "solution_prob": solution_prob.numpy(),
                                             "first_excited_state_prob": first_excited_state_prob.numpy(),
                                             "offset": offset,
                                             "time": end_time - start_time,
                                             "energy": energy.numpy() }
        #for var, value in self.samplesets["qaoa_pennylane"].items():
        #    print(var, type(value))
        return self.samplesets["qaoa_pennylane"]
        
        
    def solve_with_QAOA_Qiskit(self):
        qaoa_qiskit = QAOAQiskit(self.bqm)
        H = qaoa_qiskit.get_Hamiltonian()
        sampler = primitives.Sampler()
        labels = qaoa_qiskit.get_labels()
        qaoa = QAOA(sampler, GradientDescent(maxiter=1000))
        result = qaoa.compute_minimum_eigenvalue(H)
        best_bitstring = result.best_measurement["bitstring"]
        
        # Because the problem was encoded with Ising model, 
        # we convert the result back to QUBO: |0> -> 1 and |1> -> -1
        # This means that the measured 0s represent the solution to the problem 
        solution = dict(zip(reversed(labels), [(int(i) + 1) % 2 for i in best_bitstring]))
        self.samplesets["qaoa_qiskit"] = { "result": solution, 
                                          "info": result, 
                                          "offset": qaoa_qiskit.get_offset(), 
                                          "n_qubits": len(labels) }
        return result
    
    def get_samplesets(self):
        return self.samplesets
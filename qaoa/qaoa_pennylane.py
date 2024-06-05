import dimod
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane import qaoa
from scipy.sparse.linalg import eigsh

import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

class QAOAPennylane:

    def __init__(self, bqm, depth, n_included_terms = None) -> None:
        self.depth = depth
        if type(bqm) == dimod.BinaryPolynomial:
            self.bqm = bqm
        elif type(bqm) == dimod.BinaryQuadraticModel:
            self.bqm = bqm
        else:
            raise TypeError("BQM must be of type dimod.BinaryPolynomial or dimod.BinaryQuadraticModel")
        
        self.n_included_terms = n_included_terms
        self.variables = list(self.bqm.variables)
        self.circuit, self.probability_circuit, self.variables_to_observables, self.offset = self.bqm_to_circuit()
        self.circuit = qml.compile(self.circuit)

    def bqm_to_circuit(self):
        print("Variables:", self.variables)
        
        n_vars = len(self.variables)
        if n_vars > 21:
            raise ValueError("Currently, the number of variables must be < 22.")
        
        variables_to_qubits = dict(zip(self.variables, range(n_vars)))
        self.qubits_to_variables = {v: k for k, v in variables_to_qubits.items()}
        h = None
        if type(self.bqm) == dimod.BinaryPolynomial:
            self.bqm = self.bqm.to_spin()
            h, J, offset = self.bqm.to_hising()
        elif type(self.bqm) == dimod.BinaryQuadraticModel:
            h, J, offset = self.bqm.to_ising()
        
        self.number_of_terms = len(J) + len(h)
        if self.n_included_terms is None:
            self.n_included_terms = self.number_of_terms
        dev = qml.device('lightning.gpu', wires=n_vars)

        vars_to_obs = {v: qml.PauliZ(i) for v, i in variables_to_qubits.items()}
        coeffs, obs = [], []
        
        # Encode the linear terms
        for var, coeff in h.items():
            coeffs.append(coeff)
            obs.append(vars_to_obs[var])

        # Encode the quadratic or higher order terms
        for var, coeff in J.items():
            coeffs.append(coeff)
            obs_list = [vars_to_obs[v] for v in var]
            obs.append(Tensor(*obs_list))

        print("Number of observables:", len(obs))
        
        H_mix = qaoa.x_mixer(wires=range(n_vars))
        #H_total_cost = qml.Hamiltonian(coeffs, obs)
        H_total_cost = qml.Hamiltonian(coeffs[:self.n_included_terms], obs[:self.n_included_terms])
        
        if False:
            H_matrix = H_total_cost.sparse_matrix()
            eigenvalue, eigenvector = eigsh(H_matrix, k=1, which='SA')
            min_eigval = eigenvalue[0]
            min_eigvec = eigenvector[:, 0]
            #print("Minimum Eigenvalue:", eigenvalue[0])
            #print("Corresponding Eigenvector:", eigenvector[:, 0])
            
            #eigvals, eigenvectors = np.linalg.eigh(H_matrix)
            #min_eigval = np.min(eigvals)
            print("Minimum eigenvalue:", min_eigval)
            #min_eigvec = eigenvectors[:, np.argmin(eigvals)]
            index_1 = np.min(np.nonzero(min_eigvec == 1)[0])
            bitstring = np.binary_repr(index_1, width=n_vars)
            print("Analytic solution:")
            result = {self.qubits_to_variables[i] : int(b) for i, b in enumerate(bitstring)}
            for r in result:
                print(r, (result[r] + 1) % 2)
        
        H_costs = [qml.Hamiltonian([coeff], [ob]) for coeff, ob in zip(coeffs, obs)]
        
        def qaoa_layer(params):
            for j in range(self.n_included_terms):
                qaoa.cost_layer(params[j], H_costs[j])
            #i = 0
            #for coeff, ob in zip(coeffs, obs):
                #qaoa.cost_layer(jnp.asarray(params)[i], H_cost)
            #    if i < self.n_included_terms:
            #        H_cost = qml.Hamiltonian([coeff], [ob])
            #        qaoa.cost_layer(params[i], H_cost)
            #    i += 1
            #qaoa.mixer_layer(jnp.asarray(params)[-1], H_mix)
            qaoa.mixer_layer(params[-1], H_mix)
        
        def layers(params):
            qml.broadcast(qml.Hadamard, wires=range(n_vars), pattern="single")
            #qml.layer(qaoa_layer, self.depth, params)
            #for i in range(self.depth):
                #qaoa_layer(jnp.asarray(params)[i])
            qaoa_layer(params)

        #@jax.jit
        @qml.qnode(dev)#, interface="jax")
        def circuit(params):
            layers(params)
            return qml.expval(H_total_cost)
        
        if self.n_included_terms < 0:
            fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")([[0.5]*(len(obs) + 1)])
            fig.savefig(f"QAOA_circuit_pennylane_{n_vars}.png")
        
        @qml.qnode(dev)#, interface="jax")
        def probability_circuit(params):
            layers(params)
            return qml.probs(wires=range(n_vars))

        return circuit, probability_circuit, vars_to_obs, offset
    
    
    def get_variables_to_observables(self):
        return self.variables_to_observables
    
    def get_observables_to_variables(self):
        return {v: k for k, v in self.variables_to_observables.items()}
    
    def get_variables(self):
        return self.variables
    
    def get_n_vars(self):
        return len(self.get_variables())

    def get_circuit(self):
        return self.circuit
    
    def get_offset(self):
        return self.offset
    
    def get_probability_circuit(self):
        return self.probability_circuit
    
    def get_qubits_to_variables(self):
        return self.qubits_to_variables
    
    def get_number_of_terms(self):
        return self.number_of_terms
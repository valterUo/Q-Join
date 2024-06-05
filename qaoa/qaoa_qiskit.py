import dimod
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp

class QAOAQiskit:
    
    def __init__(self, bqm):
        self.bqm = bqm
        self.operator = self.bqm_to_hamiltonian()
        self.variables = list(self.bqm.variables)
        
        
    def get_Hamiltonian(self):
        return self.operator
    
    def get_offset(self):
        return self.offset
    
    def get_qubits_to_variables(self):
        return self.qubits_to_variables
    
    def get_variables_to_qubits(self):
        return self.variables_to_qubits
    
    def get_variables(self):
        return self.variables
    
    
    def bqm_to_hamiltonian(self):
        # labels = ["XX", "XX", "XX", "YI", "II", "XZ", "XY", "XI"]
        # coeffs = [2.+1.j, 2.+2.j, 3.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j]
        
        obs, coeffs = [], []
        self.variables = list(self.bqm.variables)
        n_vars = len(self.variables)
        self.variables_to_qubits = dict(zip(self.variables, range(n_vars)))
        self.qubits_to_variables = {v: k for k, v in self.variables_to_qubits.items()}
        
        if type(self.bqm) == dimod.BinaryPolynomial:
            h, J, self.offset = self.bqm.to_hising()
        elif type(self.bqm) == dimod.BinaryQuadraticModel:
            h, J, self.offset = self.bqm.to_ising()
        else:
            raise TypeError("BQM must be of type dimod.BinaryPolynomial or dimod.BinaryQuadraticModel")
        
        for var, coeff in h.items():
            coeffs.append(coeff)
            qubit_id = self.variables_to_qubits[var]
            label = "I"*n_vars
            label = label[:qubit_id] + "Z" + label[qubit_id+1:]
            obs.append(label)
        
        for var, coeff in J.items():
            coeffs.append(coeff)
            qubit_ids = [self.variables_to_qubits[v] for v in var]
            label = "I"*n_vars
            for qubit_id in qubit_ids:
                label = label[:qubit_id] + "Z" + label[qubit_id+1:]
            obs.append(label)
        
        return SparsePauliOp(obs, coeffs=coeffs) #.simplify()
        
    
    def get_ising_matrix(self):
        self.bqm.normalize()
        linear, quadratic, offset = self.bqm.to_ising()
        ising_model = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.SPIN)
        lin, (row, col, quad), offset, labels = ising_model.to_numpy_vectors(sort_indices=True, return_labels=True)
        dim = len(lin)
        Q = np.zeros((dim, dim))
        np.fill_diagonal(Q, lin)
        Q[row, col] = quad
        return Q, labels, offset
    
    
    def bqm_to_circuit(self, J):
        num_nodes = len(J)
        pauli_list, coeffs = [], []
        
        for i in range(num_nodes):
            # Linear terms on the diagonal
            if J[i, i] != 0:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                pauli = Pauli((z_p, x_p))
                pauli_list.append(pauli)
                coeffs.append(J[i, i])

            # Quadratic terms on the off-diagonal
            for j in range(i+1, num_nodes):
                if J[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i], z_p[j] = True, True
                    pauli = Pauli((z_p, x_p))
                    pauli_list.append(pauli)
                    coeffs.append(J[i, j])

        return SparsePauliOp(pauli_list, coeffs=coeffs).simplify()
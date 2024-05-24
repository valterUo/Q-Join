import dimod
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp

class QAOAQiskit:
    
    def __init__(self, bqm):
        if type(bqm) == dimod.BinaryPolynomial:
            self.bqm = bqm
        elif type(bqm) == dimod.BinaryQuadraticModel:
            self.bqm = bqm
        else:
            raise TypeError("BQM must be of type dimod.BinaryPolynomial or dimod.BinaryQuadraticModel")
        
        J, self.labels, self.offset = self.get_ising_matrix()
        self.operator = self.bqm_to_circuit(J)
        
    def get_Hamiltonian(self):
        return self.operator
    
    def get_labels(self):
        return self.labels
    
    def get_offset(self):
        return self.offset
        
    
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
import dimod
import pennylane as qml
from pennylane import qaoa


class QAOAPennylane:

    def __init__(self, bqm, depth) -> None:
        self.depth = depth
        if type(bqm) == dimod.BinaryPolynomial:
            self.bqm = bqm
        elif type(bqm) == dimod.BinaryQuadraticModel:
            self.bqm = bqm
        else:
            raise TypeError("BQM must be of type dimod.BinaryPolynomial or dimod.BinaryQuadraticModel")
        
        self.variables = list(self.bqm.variables)
        self.circuit, self.probability_circuit, self.variables_to_observables, self.offset = self.bqm_to_circuit()

    def bqm_to_circuit(self):
        print("Variables:", self.variables)
        
        n_vars = len(self.variables)
        if n_vars > 21:
            raise ValueError("Currently, the number of variables must be < 22.")
        
        variables_to_qubits = dict(zip(self.variables, range(n_vars)))
        
        if type(self.bqm) == dimod.BinaryPolynomial:
            h, J, offset = self.bqm.to_hising()
        elif type(self.bqm) == dimod.BinaryQuadraticModel:
            h, J, offset = self.bqm.to_ising()
        
        dev = qml.device("lightning.qubit", wires=n_vars)

        vars_to_obs = {v: qml.PauliZ(i) for v, i in variables_to_qubits.items()}
        coeffs, obs = [], []

        # Encode the linear terms
        for var, coeff in h.items():
            coeffs.append(coeff)
            obs.append(vars_to_obs[var])

        # Encode the quadratic terms or higher order terms
        for var, coeff in J.items():
            coeffs.append(coeff)
            obs_list = []
            for i in range(len(var)):
                obs_list.append(qml.PauliZ(variables_to_qubits[var[i]]))
            obs.append(qml.operation.Tensor(*obs_list)) # type: ignore

        self.H_cost = qml.Hamiltonian(coeffs, obs)
        H_mix = qml.Hamiltonian([1]*n_vars, [qml.PauliX(i) for i in range(n_vars)]) # type: ignore

        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, self.H_cost)
            qaoa.mixer_layer(alpha, H_mix)
            
        def layers(params, **kwargs):
            qml.broadcast(qml.Hadamard, wires=range(n_vars), pattern="single")
            qml.layer(qaoa_layer, self.depth, params[0], params[1])

        @qml.qnode(dev)
        def circuit(params):
            layers(params)
            return qml.expval(self.H_cost)
        
        @qml.qnode(dev)
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
    
    def get_Hamiltonian(self):
        return self.H_cost
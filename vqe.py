import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, transpile, assemble, execute
from qiskit.algorithms.optimizers import COBYLA
import pickle
from key import tok
from qiskit import IBMQ #2
from qiskit.providers.aer.noise import NoiseModel

'''
Author: Maria Gabriela Jordão Oliveira

The goal of this script is to implement the Variational Quantum Eigensolver (VQE) for
the two-site Heisenberg model. The Hamiltonian is given by:
H = J (S1x S2x + S1y S2y + S1z S2z)
where J is the coupling constant and S1 and S2 are the spin operators for the two sites.

The results are saved in pickle files.
'''
# Save account
#IBMQ.save_account(tok, overwrite=True)

# Load account
IBMQ.load_account()

#Get your provider
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

# Choose backend - in the article we used ibm_brisbane ans ibmq_qasm_simulator
backend_device = provider.get_backend('ibm_brisbane') 
# QASM simulator
backend = Aer.get_backend('qasm_simulator')

# Simulator with brisbane noise model
noise_model = NoiseModel.from_backend(backend_device)


# Constants
random_seed=42
dimension=2
NUM_SHOTS = 20000

# List to save the energy values in each iteration
energy_values = []

def qml_circuit(params,measure):
    """
    This function takes the parameters of our variational form as arguments and returns the corresponding quantum circuit.
    """
    
    # Create a quantum circuit with dimension qubits
    qr = QuantumRegister(dimension, name="q")
    cr = ClassicalRegister(dimension, name='c')
    qc = QuantumCircuit(qr, cr)
    
    # Initialization
    j=0
    for i in range(dimension):
        qc.rx(params[j], qr[i]) 

        qc.ry(params[j+1], qr[i])
        
        
        qc.rz(params[j+2], qr[i])

        j+=3
        
    # Entanglement layer    
    for i in range(dimension):
        qc.cx(qr[i], qr[(i+1)%dimension])
  

    # 0 = x, 1 = y, 2 = z
    # Measurement layer - measure along x
    if measure==0:
        for i in range(dimension):
            qc.ry(np.pi/2, qr[i])          
            qc.rx(np.pi, qr[i])
            qc.measure(qr[i], cr[i])
            
    # Measurement layer - measure along y
    if measure==1:
        for i in range(dimension):
            qc.ry(np.pi/2, qr[i]) 
            qc.rx(np.pi, qr[i])
            qc.rz(np.pi/2, qr[i]).inverse()
            qc.measure(qr[i], cr[i])
            
    # Measurement layer - measure along z
    if measure==2:
        qc.measure(qr, cr)
    # print(qc.draw())
    return qc

def objective_function(params):
    # Calculate the cost as the energy value, i.e. we want to minimize the energy value
    
    Energy_meas = [] # list to save the energy values for each measurement
    for measure_circuit in [0, 1, 2]:
        # run the circuit with the selected measurement and get the number of samples that output each bit value
        qc = qml_circuit(params,measure_circuit)
        counts = execute(qc, backend=backend, shots=NUM_SHOTS, noise_model=noise_model).result().get_counts()
        
        # calculate the probabilities for each computational basis
        probs = {}
        for output in ['00','01', '10', '11']:
            if output in counts:
                probs[output] = counts[output]/NUM_SHOTS
            else:
                probs[output] = 0
        # Save the energy values (of one measurement)
        Energy_meas.append( probs['00'] - probs['01'] - probs['10'] + probs['11'] )
    # Save the energy values (of all measurements)
    energy_values.append(np.sum(np.array(Energy_meas)))
    # Return the expectation value 
    return np.sum(np.array(Energy_meas))


# Initialize the COBYLA optimizer
optimizer = COBYLA(maxiter=150, tol=0.1)

# Create the initial parameters (noting that our single qubit variational form has 3 parameters)
params = np.random.rand(3*dimension)

# Run the optimization
#ret = optimizer.minimize(fun=objective_function, x0=params)
ret = optimizer.optimize(num_vars=3*dimension, objective_function=objective_function, initial_point=params)
# Save the results
save = {}
save = {}
save['params'] = ret[0]
save['Energy_found'] = ret[1]
save['Energy_values'] = energy_values

# save the results with pickle
with open('vqe_noise_test.pkl', 'wb') as f:
    pickle.dump(save, f)

print("Energy found:", ret[1])

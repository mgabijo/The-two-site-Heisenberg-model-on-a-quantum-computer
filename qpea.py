import numpy as np
from qiskit import *
from key import tok
from qiskit import IBMQ 
import pickle
'''
Author: Maria Gabriela Jordão Oliveira

The goal of this script is to implement the Quantum Phase Estimation Algorithm (QPEA) for
the two-site Heisenberg model. The Hamiltonian is given by:
H = J (S1x S2x + S1y S2y + S1z S2z)
where J is the coupling constant and S1 and S2 are the spin operators for the two sites.

The results are saved in pickle files.
'''

# Save account
IBMQ.save_account(tok, overwrite=True)

# Load account
IBMQ.load_account()

#Get your provider
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

# Choose backend - in the article we used ibm_brisbane and ibmq_qasm_simulator
# Please comment the backend you are not using
#backend = provider.get_backend('ibm_brisbane') 
backend = Aer.get_backend('qasm_simulator')

# Define the number of qubits and shots 
# 5 qubits = 3 counting qubits + 2 state qubits
qubits = 5
shots = 20000


# Define QFTDagger
def qft_dagger(qc, n):
    """n-qubit QFTdagger the first n qubits in circuit qc
        input qc: QuantumCircuit
        input n: int - number of qubits to be QFTdaggered
    """
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)


def get_phase(t,state):
    '''
    Function that implements the Quantum Phase Estimation Algorithm (QPEA) for the two-site Heisenberg model.
    Input: 
        t : time parameter for the time evolution
        state: 1 or -1 - initial state of the system (1/sqrt(2)(|00> + |11>) or 1/sqrt(2)(|00> - |11>))
    Output:
        answer: dictionary with the results of the measurements
    '''
    # Create Quantum Circuit with the necessary qubits - 2 qubits for the state and 3 for the counting
    qpe = QuantumCircuit(qubits, qubits-2)

    # Inicialize the state qubits in the desired state
    qpe.initialize(params=[0, 1/np.sqrt(2),state*1/np.sqrt(2), 0],qubits=[qubits-2,qubits-1])


    # Create superposition in the counting qubits - Hadamard gates
    for qubit in range(qubits-2):
        qpe.h(qubit)

    # Apply controlled unitary operations
    # the repetitions variable is used to control the number of repetitions of each controlled unitary operation
    repetitions = 1
    for counting_qubit in range(qubits-2):
        for i in range(repetitions):
            # Definition of each controlled unitary operation
            qpe.cx(qubits-2,qubits-1)
            qpe.crx(2*t,counting_qubit,qubits-2)
            qpe.crz(2*t,counting_qubit,qubits-1)
            qpe.cx(qubits-2,qubits-1)
            qpe.cy(qubits-2,qubits-1)
            qpe.cry(2*t,counting_qubit,qubits-2)
            qpe.cy(qubits-2,qubits-1)
        repetitions *= 2

    qpe.barrier()

    # Apply inverse QFT
    qft_dagger(qpe, qubits-2)

    # Measure
    qpe.barrier()
    for n in range(qubits-2):
        qpe.measure(n,n)
   
    # If you want to see the circuit, uncomment the line below
    #print(qpe.draw())
    
    # Execute the circuit and get the results (counts)
    answer = (
                execute(qpe, backend=backend, shots=shots,) #optimization_level=3) # ignore optimization_level to run on simulator
                .result()
                .get_counts()
            )
    return answer

# Run the QPEA for the two states
for state in [-1,1]:
    # Create a list to save the time parameter and the results of the measurements
    tau=[]
    results=[]
    # Run the QPEA for 30 different values of the time parameter - time evolution
    for t in np.arange(0,2*np.pi,2*np.pi/30):
        t = float(t)
        tau.append(t)
        results.append(get_phase(t,state))
        
        # Define the name of the run
        run = 'simulator'
        # Save the results in pickle files
        # Note that this step is inside the loop to avoid losing the results if the execution is interrupted
        if state==1:
            open_file = open(f"tau_qpea_{run}_1.pkl", "wb")
            pickle.dump(tau,open_file)
            open_file.close()
            open_file = open(f"results_qpea_{run}_1.pkl", "wb")
            pickle.dump(results,open_file)
            open_file.close()
        if state==-1:
            open_file = open(f"tau_qpea_{run}.pkl", "wb")
            pickle.dump(tau,open_file)
            open_file.close()
            open_file = open(f"results_qpea_{run}.pkl", "wb")
            pickle.dump(results,open_file)
            open_file.close()
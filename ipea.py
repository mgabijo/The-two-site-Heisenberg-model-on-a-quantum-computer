from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
import pickle
from key import tok
from qiskit import IBMQ #2
import numpy as np
'''
Author: Maria Gabriela Jordão Oliveira

The goal of this script is to implement the Iterative Quantum Phase Estimation Algorithm (IPEA) for
the two-site Heisenberg model. The Hamiltonian is given by:
H = J (S1x S2x + S1y S2y + S1z S2z)
where J is the coupling constant and S1 and S2 are the spin operators for the two sites.

The results are saved in pickle files.
'''

# Save account
IBMQ.save_account(tok, overwrite=True)

# Load account
IBMQ.load_account()

# Choose backend - in the article we used ibm_brisbane ans ibmq_qasm_simulator
# Please comment the backend you are not using
#backend = provider.get_backend('ibm_brisbane') 
backend = Aer.get_backend('qasm_simulator')

# Define number of shots 
shots = 20000

# Auxiliary function to get the most probable value for the bit
def most_probable(dic):
    m = list(dic)[0]
    for n in list(dic):
        if dic[n]>dic[m]:
            m = n
    return m

def get_bit(t,
        estado,
        iteration,
        rotation,
        cbits
    ):
        '''
        Function to retrieve a single bit from the IPEA.
        Input:
            t : time parameter for the time evolution
            iteration: number of the iteration, bit number
            rotation: angle of the correction rotation
            cbits: number of bits  
        '''
        exponent = 2 ** (cbits-iteration-1)

        ancilla=[0]
        nq = 3    # number of qubits
        m = 1    # number of classical bits
        q = QuantumRegister(nq,'q') # quantum register with nq qubits
        c = ClassicalRegister(m,'c') # classical register with m bits
        QC = QuantumCircuit(q,c) # quantum circuit with quantum and classical registers
        
        # Initialize the state register
        QC.initialize(params=[0, 1/np.sqrt(2),estado*1/np.sqrt(2), 0],qubits=[1,2]) 
    
        # Initialize the ancilla qubit in the |+> state
        QC.h(ancilla)
        
        # add the correction rotation
        QC.p(rotation, ancilla)

        # add the controlled unitary operations
        # the number of repetitions of each controlled unitary operation is given by the exponent
        for _ in range(int(exponent)):
            QC.cx(1,2)
            QC.crx(2*t,0,1)
            QC.crz(2*t,0,2)
            QC.cx(1,2)
            QC.cy(1,2)
            QC.cry(2*t,0,1)
            QC.cy(1,2)

        # add H gate
        QC.h(ancilla)
        
        # Measure the ancilla qubit
        QC.measure(ancilla, c[0])

        # Execute the circuit and get the counts
        counts = (
                execute(QC, backend=backend, shots=shots)
                .result()
                .get_counts()
            )
            

        return counts

# Function to implement the IPEA
def get_phase(t,state,cbits):
    '''
    Implementation of the Iterative Quantum Phase Estimation Algorithm (IPEA) for the two-site Heisenberg model
    Input:
        t : time parameter for the time evolution
        state : 1 or -1 - initial state of the system (1/sqrt(2)(|00> + |11>) or 1/sqrt(2)(|00> - |11>))
        cbits : number of bits to be retrieved
    '''
    # Result list contains the results of each iteration, i.e. the bits retrieved
    result = []
    
    # Phase factor
    phase = -2 * np.pi
    
    # The initial factor is 0
    factor = 0
    
    # Iterate over the number of bits to be retrieved
    for it in range(cbits):
        # Get the correction phase
        inv_phase = phase * factor
        
        # Get the result of the measurement of the ancilla qubit 
        # only one bit is retrieved
        dic_bit = get_bit(t,state,it,inv_phase,cbits)

        # Determine the most probable value for the bit
        m=most_probable(dic_bit)
        # Save the bit in the result list
        result.append(int(m))
         
        # Update the factor    
        if m == "1":
            factor += 1 / 2  # add the phase factor
        factor = factor / 2  # shift each towards one weight righ
    
    # need to reverse as LSB is stored at the zeroth index and
    # not the last
    result = result[::-1]

    # find decimal phase
    dec = 0
    weight = 1 / 2
    for k in result:
        dec += (weight) * k
        weight /= 2
    return dec

for state in [-1,1]:
    # Execute the algorithm for both states
    
    # Create a list to save the time parameter and the results of the measurements
    tau=[]
    results=[]
    
    # Run the IPEA for 30 different values of the time parameter - time evolution
    for t in np.arange(0,2*np.pi,2*np.pi/30):
        t=float(t)
        tau.append(t)
        results.append(get_phase(t,
        state,
        3
    ))
        # Define the name of the run
        run = 'test'
        # Save the results in pickle files
        # Note that this step is inside the loop to avoid losing the results if the execution is interrupted
        if state==1:
            open_file = open(f"tau_ipea_{run}_1.pkl", "wb")
            pickle.dump(tau,open_file)
            open_file.close()
            open_file = open(f"resposta_ipea_{run}_1.pkl", "wb")
            pickle.dump(results,open_file)
            open_file.close()
        if state==-1:
            open_file = open(f"tau_ipea_{run}.pkl", "wb")
            pickle.dump(tau,open_file)
            open_file.close()
            open_file = open(f"resposta_ipea_{run}.pkl", "wb")
            pickle.dump(results,open_file)
            open_file.close()
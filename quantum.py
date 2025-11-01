import numpy as np
import math
import random

j = complex(0,1)


### some quantum gates, which are unitary matrices
hadamard = np.array([[1, 1],
                     [1, -1]]) * 1/math.sqrt(2)

cnot = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

cz = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,-1]
])

cs = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,j]
])

pauli_x = np.array([
    [0,1],
    [1,0]
])

pauli_z = np.array([
    [1,0],
    [0,-1]
])

identity = np.array([[1, 0],
                     [0, 1]])

s_dagger = np.array([[1, 0],
                    [0, -j]])

c_half_not = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0.5 + 0.5*j,0.5 - 0.5*j],
                       [0,0,0.5 - 0.5*j,0.5 + 0.5*j]])

sx = np.array([[(1+j)/2,(1-j)/2],[(1-j)/2,(1+j)/2]])

### A quantum state of b qubits is a "vertical" array, i.e. a column vector, of length 2^qubits, 
### where each element is a complex number called amplitude, 
### and such that the 2-norm of the vector is 1
###
### in order to initialize a state,
### instead of explicitly writing the amplitude of each case, we give a list of non-zero entries
###
### Example: we want to initialize 3 qubits all to 0
### we write:
### from_amplitudes([(0,1)],3)
### which is equivalent to:
### np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
### that is, we have amplitude 1 in position 0, and amplitude 0 in all other positions
def from_amplitudes(v,bits):
    state = [[0.0] for i in range(1<<bits)]
    for i,a in v:
        state[i][0] = a
    return np.array(state)

### sometimes we want to know, for each possible measurement outcome, what is its probability
def probabilities(state):
    return [round(float(np.abs(x[0])*np.abs(x[0])),7) for x in state]

### this function computes the kronecker (or tensor) product of the given matrices
def combine(list):
    if len(list) == 1:
        return list[0]
    r = np.kron(list[0],list[1])
    for m in list[2:]:
        r = np.kron(r,m)
    return r


### suppose we want to apply a specific gate on some qubits of our state
### let b = len(bits), then gate must be a 2^b x 2^b unitary matrix
### our state may containg strictly more qubits, so we need to compute a larger matrix, that applies the gate on the correct qubits and does nothing on the others
### compute_u computes the larger matrix,
### it first permutes the qubits, so that the b required bits are in the first positions and in the correct order,
### then applies gate  ⊗ identity ⊗ ... ⊗ identity, then undoes the permutation
def compute_u(gate,bits,n):
    # if we need to apply some matrix on the i-th qubit,
    # we compute identity ⊗ identity ⊗ ... ⊗ identity ⊗ matrix ⊗ identity ⊗ ... ⊗ identity
    # where the matrix is in the i-th position
    if len(bits) == 1:
        return combine([identity if bits[0]!=i else gate for i in range(0,n)])
    # otherwise, things are a bit more complicated
    # it two qubits are adjacent (i.e. they are the ith and the (i+1)-th qubit for some i) we can use a similar trick as in the previous case
    # if not, we use some swaps to bring the desired qubits to be adjacent, then we do things as in the easy case, then we undo the swaps
    else:
        # bring the bits to the first len(bits) positions
        swap_inv = np.zeros((1<<n,1<<n))
        for i in range (0,1<<n):
            v = [0]*n
            for j in range(0,n):
                v[j] = (i>>(n-1-j)) % 2
            w = [0]*n
            for j in range(0,len(bits)):
                w[bits[j]] = 1
            z = [0]*n
            for j in range(0,len(bits)):
                z[j] = v[bits[j]]
            k = len(bits)
            for j in range(0,n):
                if w[j] == 0 :
                    z[k] = v[j]
                    k += 1
            new_pos = 0
            for j in range(0,n):
                new_pos = 2*new_pos + z[j]
            swap_inv[i][new_pos] = 1
        ops = [gate]
        for i in range(0,n-len(bits)):
            ops.append(identity)
        # then apply the matrix on the first qubits and indentity on all the others
        matrix = combine(ops)
        # then bring qubits in their original position
        swap = swap_inv.transpose()
        # operations are applied from right to left
        return np.matmul(swap_inv,np.matmul(matrix, swap))

### this function takes a state, a gate to apply, and a list of qubits on which to apply the gate, and applies the gate to such qubits, by using compute_u
def apply(state,bits,gate):
    total_bits = (len(state)-1).bit_length()
    return np.matmul(compute_u(gate,bits,total_bits),state)


### measure a single qubit in the standard basis
def measure(state, bit):
    bits = (len(state)-1).bit_length()
    # first, compute the probability that a specific qubit, if measured, gives 0 or 1
    probs = probabilities(state)
    prob_0 = 0
    prob_1 = 0
    for i in range(0,len(probs)):
        if (i >> (bits-bit-1))&1 == 0:
            prob_0 += probs[i]
        else:
            prob_1 += probs[i]
    # sample the measured qubit according to the computed probability
    result = int(random.random() >= prob_0)
    prob = prob_0 if result == 0 else prob_1
    # now restrict the state to the correct half
    new_state = []
    for i in range(0,len(probs)):
        if (i >> (bits-bit-1))&1 == result:
            new_state.append([state[i][0] / math.sqrt(prob),6])
    return result,np.array(new_state)

### print entries with non-zero amplitude
def print_state(state):
    print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
    total_bits = (len(state)-1).bit_length()

    for i in range(len(state)):
        x = state[i][0]
        # round amplitudes nicely and hide the imaginary part if not present
        x = np.real_if_close(complex(round(x.real,6),round(x.imag,6)))
        if x != 0:
            print("{:0{}b} {: g}".format(i,total_bits,x))
    print("____________________")

### Example: GHZ
### The protocol works as follows:
###
### There are 3 players, there are 3 qubits, each players holds one qubit.
### By sampling from 3 qubits there are 8 possibilities, 000, 001, 010, 011, 100, 101, 110, 111.
### Quantum says that, if we take the sum of the squares of the moduli of the 8 amplitudes, we should get 1. I.e., the 2-norm is 1.
###
### The three players start in the 0 0 0 state.
### So we put amplitude 1 to be in the case 000, and 0 in all the others.
### We represent the state with an array of size 8, where the first position represents 000 and the last represents 111.
state = from_amplitudes([(0,1)],3)
### this is equivalent to
### state = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
### note: the state is a "vertical" array, i.e. a column vector
###
### In the actual protocol, at the beginning, nodes would agree on one of the three to be the leader,
### the others would send their qubit to the leader,
### and hence the leader can perform an operation that acts on all three qubits at the same time.
###
### Note: at this point the inputs are still unknown.
### The goal at this point is to reach the state in which
### we have 1/sqrt(2) probability of being in the state 000 and 1/sqrt(2) of being in the state 111
### (note that the squares sum to 1).
### this is called "GHZ state", and it is like a Bell state, but with one more qubit
###
### The following operations would be done by the leader.
### In order to obtain the aforementioned state, the leader needs to do the following:
### 1. apply the hadamard gate
### If we think about what this first step achieves classically, it goes from the state in which the first bit is 0 to the state in which the first bit is random
state = apply(state,[0],hadamard)
###
### 2. Then we apply the CNOT gate on the first two qubits and nothing on the third.
### CNOT stands for controlled not, and what it would do "classically" is to apply NOT to the second bit only if the first bit is 1,
### Hence, if the first bit is 0, the second stays 0, while if the first bit is 1, the second becomes 1
state = apply(state,[0,1],cnot)
###
### We do exactly the same thing between the second and the third qubit
### Note, [0,2] instead of [1,2] would also work
state = apply(state,[1,2],cnot)
###
### If we now print the state, we get that the first entry (000) and the last entry (111) have values 1/sqrt(2)
#print_state(state)
### The leader would now send the qubits to the other players, going back to the case in which each player has a single qubit.
###
### Now, players receive inputs.
### They do not communicate between each other anymore,
### so they can only apply matrices on their own qubits.
###
### Let's say that the first player has input 0.
### The protocol says that, in such a case, the first player has to make a measurement in the X basis
### what this means is that we need to rotate the qubit in some way and then measure
### we can achieve this by first applying hadamard and then performing a standard measurement
state = apply(state,[0],hadamard)
###
### Now, let's say that the second player has input 1
### The protocol says that, in this case, the second player must measure in the Y basis
### this is achieved by first applying the S_Dagger gate, and then hadamard, and then measuring
state = apply(state,[1],s_dagger)
state = apply(state,[1],hadamard)
###
### Let's say that the third player has also input 1
### It does exactly the same as the second player, but now acting on the third qubit
state = apply(state,[2],s_dagger)
state = apply(state,[2],hadamard)
###
### We did not actually measure anything until now, and if we print the state we see the possible outcomes that we would get by measuring everything
### If we now print the state, we have non-zero probability of being in the states 001 010 100 111
### so the number of output 1 bits is always odd
### Which is correct according to the GHZ rules, because the OR of the input bits is 1
#print_state(state)




#############
### quantum teleportation
### suppose 1st qubit has 1/3 prob of being 0 and 2/3 prob of being 1
### so our initial state has amplitude sqrt(1/3) for 000 (0) and sqrt(2/3) for 100 (4)
state = from_amplitudes([(0,math.sqrt(1.0/3)),(4,math.sqrt(2.0/3))],3)
### then, suppose Alice and Bob share a Bell pair (qubits 1 and 2)
### which is a 2-qubit state that, if measured, gives 00 and 11 with equal probability
### this can be obtained by applying hadamard on the first qubit (which has the effect of making the first qubit random (if measured))
### and applying cnot
state = apply(state,[1],hadamard)
state = apply(state,[1,2],cnot)
###
### the previous part was the initialization, bit 0 needs to be teleported, and bits 1 (held by Alice) and 2 (held by Bob) are a Bell pair
### the teleportation protocol now does the following:
### cnot on 0 and 1
state = apply(state,[0,1],cnot)
### hadamard on 0
state = apply(state,[0],hadamard)
### Alice measures all its qubits (0 and 1)
### note: the result is that Alice has no qubits at all, and has only 2 standard bits,
### since the measure function, when measuring the qubit in position 0, destroys it, qubit 1 gets position 0
### so we measure on position 0 twice
b1,state = measure(state,0)
b2,state = measure(state,0)
### now Alice sends the two bits b1 and b2 to Bob
### Note: old qubit 2 is now qubit 0
### Bob applies the following protocol: if b2 is 1, apply the pauli_x gate
### then, if b1 is 1, apply the pauli_z gate
if b2:
    state = apply(state,[0],pauli_x)
if b1:
    state = apply(state,[0],pauli_z)
### if we now print the state, we see that we have sqrt(1/3) of getting 0, and sqrt(2/3) of getting 1, which is the initial state of Alice's qubit
#print_state(state)


#######
### Entanglement Swapping 
### suppose Alice has a qubit that has 1/3 prob of being 0 and 2/3 prob of being 1
### we want to teleport it to Bob, via a repeater, Charlie
### all other qubits are initialized to 0
state = from_amplitudes([(0,math.sqrt(1.0/3)),(8,math.sqrt(2.0/3))],4)
### Alice has another qubit, and cnot is applied in order to reach a state in which qubits 0 and 1, when measured, give the same bit
state = apply(state,[0,1],cnot)
### Bob has a Bell pair (qubits 2 and 3)
state = apply(state,[2],hadamard)
state = apply(state,[2,3],cnot)
### Alice sends its qubit 1 to Charlie
### and Bob sends its qubit 2 to Charlie
### Charlie applies cnot on qubits [1,2] and then hadamard on 1
state = apply(state,[1,2],cnot)
state = apply(state,[1],hadamard)
### then Charlie measures qubits 1 and 2
b1,state = measure(state,1)
b2,state = measure(state,1)
### the outcome of the measurement is sent to Bob, which applies the same protocol of standard teleportation
### note: old qubit 2 is now qubit 0
if b2:
    state = apply(state,[1],pauli_x)
if b1:
    state = apply(state,[1],pauli_z)
### we see that, if we measure, we get 00 with probability sqrt(1/3) and 11 with probability sqrt(2/3)
#print_state(state)


#############
### we can always replace "measurement + application of matrices conditioned on the measurement outcome" with "quantum operations + measurement only at the end".
### While this would be useless for quantum teleportation, since we would then have to send a qubit, this example just shows that it is possible
###
### quantum teleportation circuit with deferred measurement
### 1st qubit has 1/3 prob of being 0 and 2/3 prob of being 1
state = from_amplitudes([(0,math.sqrt(1.0/3)),(4,math.sqrt(2.0/3))],3)
### bits 1 and 2 are a Bell pair
state = apply(state,[1],hadamard)
state = apply(state,[1,2],cnot)
### cnot on 0 and 1
state = apply(state,[0,1],cnot)
### hadamard on 0
state = apply(state,[0],hadamard)
### until now, nothing changed
### but now we do not measure
### we sent qubit 1 from Alice to Bob
###
### now, instead of applying pauli_x if the bit is 1, we apply cnot
### note: pauli_x is "not", so applying pauli_x if a bit is 1 means applying "not" only if the bit is 1, which is what cnot does
### controlled x on 1,2
state = apply(state,[1,2],cnot)
### similarly, instead of applying pauli_z if the bit is 1, we apply cz, which is a controlled pauli_z
### controlled z on 0,2
state = apply(state,[0,2],cz)
### now we do the measurements that we have deferred
### measure 0 and 1 (after 1st measure, 1 becomes 0)
b1,state = measure(state,0)
b2,state = measure(state,0)
#print_state(state)



### Grover Search
### there is some unknown bitstring x for which some function f satisfies f(x) = 1 and f(y) = 0 for all y != x
### first, we define a function f' that satisfies f'(x) = -1 and f'(y) = 1 for all y != x
### in the following example, we just assume to have a matrix that is obtained by starting from I and replacing the x-th 1 with -1
### we will assume that x = 010 = 2
bits = 3
Uf = np.identity(1<<bits)
Uf[2][2] = -1
### in the following, the fact that x = 2 will be unknown to the algorithm, and Uf will be used as a blackbox
### we will be able to recover that x = 010 = 2 in roughly sqrt(2^bits) operations
###
### Grover uses a special operator, called diffusion operator
### it applies hadamard on all qubits
### then it applies a matrix that has 1 in [0,0] and -1 in the rest of the diagonal
### then hadamard on all qubits again
###
### what the diffusion operator D achieves is the following:
### assume all amplitudes are real, then D replaces each amplitude x with (2 (average amplitude) - x)
U0 = -np.identity(1<<bits)
U0[0][0] = 1
h = compute_u(hadamard,[0],bits)
for i in range(1,bits):
    h = apply(h,[i],hadamard)
D = np.matmul(np.matmul(h,U0),h)
###
### Now we are ready to run the algorithm
### We initialize the state to the uniform superposition
### so we first we initialize it to 000..000
state = from_amplitudes([(0,1)],bits)
state[0] = 1
### and then we apply hadamard on each qubit
### the effect is, intuitively, to randomize each qubit
for i in range(0,bits):
    state = apply(state,[i],hadamard)
###
### then, for pi/4 sqrt(2^bits) times, we do the following
for i in range(0,round(math.pi/ 4 * math.sqrt(1<<bits))):
    ### we apply our function f'
    ### which changes the sign of the amplitude of the correct entry, without changing anything else
    state = apply(state,range(bits),Uf)
    ### then, we apply the diffusion operator D
    ### what this does is to replace each amplitude x with (2 (average amplitude) - x)
    ### Since the correct value has negative amplitude, what this achieves is to make the correct amplitude slightly bigger and all the others slightly smaller
    state = apply(state,range(bits),D)
### after pi/4 sqrt(2^bits) steps we obtain that the correct value has amplitude ~1 and all the others have amplitude ~0
### by continuing, we would actually make things worse!
#print_state(state)



########
### some quantum weirdness
### imagine Alice and Bob holding 2 qubits each
### we initialize the state to the following: amplitude 1/2 for each of the states 0000 0011 1100 1111
s = from_amplitudes([(0,1.0)],4)
s = apply(s,[0],hadamard)
s = apply(s,[3],hadamard)
s = apply(s,[0,1],cnot)
s = apply(s,[3,2],cnot)
#print_state(s)
### Alice and Bob send their qubits 1 and 2 to Charlie,
### which just changes the sign of 1111,
### and then sends the qubits back
s = apply(s,[1,2],cz)
#print_state(s)
### Alice and Bob undo the cnot on qubits 1 and 2, by reapplying cnot
### so we reach the state:
### 0000 with amplitude 1/2
### 0001 with amplitude 1/2
### 1000 with amplitude 1/2
### 1001 with amplitude -1/2
s = apply(s,[0,1],cnot)
s = apply(s,[3,2],cnot)
#print_state(s)
###
### we now discard qubits 1 and 2, and we can because if we measure them they are deterministically 0, so this measurement does not affect the rest of the state
b,s = measure(s,1)
b,s = measure(s,1)
### in fact, now the current state is:
### 00 with amplitude 1/2
### 01 with amplitude 1/2
### 10 with amplitude 1/2
### 11 with amplitude -1/2
### which is the same as before, but where we removed qubits 1 and 2
#print_state(s)
### Alice applies hadamard on her qubit
### the negative sign of 11 makes it so that the only two possible measurements after this operation are 00 and 11
s = apply(s,[0],hadamard)
#print_state(s)
###
### However! Imagine if Charlie did not flip the sign of 1111, and just sent the qubits back directly
### In that case, now we would get that the two possible measurements are 00 and 01
### so Alice and Bob have probability 1/2 to know if Charlie did something or not
### The curious fact is that the bits given back by Charlie are first cnotted and then measured, which looks like a useless operation
### In particular, it looks like that bits 0 and 3 are never affected by what Charlie does
### But this is false, because if Charlie does something then the measurement gives 00 or 11, and if Charlie does nothing then the measurement gives 00 or 01





########
### quantum symmetry breaking
### first, prepare 2 bell pairs, and exhange half pairs
s = from_amplitudes([(0,1.0)],4)
s = apply(s,[0],hadamard)
s = apply(s,[3],hadamard)
s = apply(s,[0,2],cnot)
s = apply(s,[3,1],cnot)
#print_state(s)
# apply cnot, using the own bit as control 
s = apply(s,[0,1],cnot)
s = apply(s,[3,2],cnot)
#print_state(s)
# then apply half cnot in the opposite direction
s = apply(s,[1,0],c_half_not)
s = apply(s,[2,3],c_half_not)
#print_state(s)
# then apply half not on the received qubits
s = apply(s,[1],sx)
s = apply(s,[2],sx)
print_state(s)
# the result is that exactly one node will measure 10 or 01, and exactly one will measure 00 or 11
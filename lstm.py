import numpy as np

"""
LSTM implementation solving the XOR problem.
Assumes binary values, reduced sigmoid/tanh functions.
"""

def S(x): return x > 0
def tanh(x): return (x > 0) ^ (x < 0)

Wf, bf = np.array([ 0, -1]), np.array([1])
Wi, bi = np.array([ 0,  1]), np.array([0])
WC, bC = np.array([-1,  0]), np.array([1])
Wo, bo = np.array([ 0,  0]), np.array([1])

def lstm(xt, hp, Cp):
    f  = S(Wf @ [hp, xt] + bf)     # ¬X
    i  = S(Wi @ [hp, xt] + bi)     # X
    Ĉt = tanh(WC @ [hp, xt] + bC)  # ¬Y
    Ct = f * Cp + i * Ĉt           # (¬X ∧ Y) ∨ (X ∧ ¬Y)
    o  = S(Wo @ [hp, xt] + bo)     # True
    ht = o * tanh(Ct)              # (¬X ∧ Y) ∨ (X ∧ ¬Y)
    return int(Ct.item()), int(ht.item())

# We assume xt ∈ {0, 1} and Cp ∈ {0, 1}
for xt in np.arange(2):
    for Cp in np.arange(2):
        hp = Cp # We assume hp = Cp
        Ct, ht = lstm(xt, hp, Cp)
        print(xt, Cp, Ct, ht)
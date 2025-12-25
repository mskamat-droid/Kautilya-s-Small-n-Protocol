#This includes the core loss functions and minimax grid evaluations.
Appendix A: Core Replication Code
Provides executable Python code for the baseline loss function (Equation 2), comparative mechanisms (Table 2), and minimax evaluations. Helps to verify all reported losses and optima The following Python code replicates the baseline loss calculations (Table 2) and supports minimax evaluations. It defines the loss function and computes values for n=1–6 under specified parameters.
Python
import numpy as np
import itertools

def loss(n, pi=0.05, q=0.75, C_FP=3000, C_FN=2500, delta=10, gamma=15, rho=0.1, Cc=1000):
    n = int(round(n))
    if n < 1:
        return np.inf
    fp = (1 - pi) * ((1 - q) ** n) * C_FP
    fn = pi * (1 - q ** n) * C_FN
    admin = n * delta
    links = n * (n - 1) / 2
    friction = gamma * links
    coll = Cc * (rho ** links)
    return fp + fn + admin + friction + coll

# Baseline examples (Table 2)
print("Baseline Kautilyan n=3:", loss(3))
print("Frictionless n=4 (gamma=0):", loss(4, gamma=0))

# Minimax grid (supporting Appendix G)
gammas = np.linspace(10, 25, 16)
rhos = np.linspace(0.05, 0.15, 11)
pis = np.linspace(0.01, 0.35, 15)
qs = np.linspace(0.7, 0.9, 5)

worst_case = {}
for n in range(1, 7):
    max_loss = max(loss(n, pi=p, q=q, gamma=g, rho=r)
                   for g in gammas for r in rhos for p in pis for q in qs)
    worst_case[n] = max_loss
    print(f"n={n} worst-case loss: {max_loss:.0f}")

# Output verifies n=3 minimises worst-case loss

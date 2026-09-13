import numpy as np

# ---------------------------------------------------------
# Big-M Simplex Method
# Problem:
# Maximize Z = 3x1 + 2x2
# Subject to:
#   x1 + x2 >= 4
#   2x1 + x2 <= 5
#   x1, x2 >= 0
# ---------------------------------------------------------

M = 1_000_000.0

# Columns: x1, x2, s1, s2, a1
# Constraint 1: x1 + x2 - s1 + a1 = 4
# Constraint 2: 2x1 + x2      + s2 = 5

A = np.array([
    [1, 1, -1, 0, 1],
    [2, 1,  0, 1, 0]
], dtype=float)

b = np.array([4, 5], dtype=float)

variables = ["x1", "x2", "s1", "s2", "a1"]

# Objective: Max Z = 3x1 + 2x2 - M*a1
C = np.array([3, 2, 0, 0, -M], dtype=float)

# Initial basic variables are a1 and s2.
basis = [4, 3]


def print_tableau(tableau, basis, iteration):
    print(f"\n--- Big-M Iteration {iteration} ---")
    print("Basis:", [variables[i] for i in basis])
    print("     " + " ".join(f"{v:>10}" for v in variables) + " |        RHS")
    for i, row in enumerate(tableau[:len(basis)]):
        name = variables[basis[i]]
        print(f"{name:>4} " + " ".join(f"{x:10.3f}" for x in row[:-1])
              + f" | {row[-1]:10.3f}")
    print("Zj  " + " ".join(f"{x:10.3f}" for x in tableau[-2, :-1])
          + f" | {tableau[-2,-1]:10.3f}")
    print("Cj-Zj " + " ".join(f"{x:7.3f}" for x in tableau[-1, :-1]))


# Tableau contains constraint rows followed by Zj row and Cj-Zj row.
tableau = np.hstack([A, b.reshape(-1, 1)])

iteration = 0

while True:
    # Calculate Zj from current basis costs.
    cb = C[basis]
    Zj = cb @ tableau[:len(basis), :]
    Cj_minus_Zj = np.append(C, 0) - Zj

    tableau_with_objective = np.vstack([
        tableau,
        Zj,
        Cj_minus_Zj
    ])

    print_tableau(tableau_with_objective, basis, iteration)

    # For maximization, positive Cj-Zj means improvement is possible.
    entering_candidates = np.where(Cj_minus_Zj[:-1] > 1e-9)[0]

    if len(entering_candidates) == 0:
        break

    entering = entering_candidates[np.argmax(Cj_minus_Zj[entering_candidates])]

    # Ratio test: RHS / positive pivot-column coefficient.
    ratios = []
    for i in range(len(basis)):
        if tableau[i, entering] > 1e-9:
            ratios.append((tableau[i, -1] / tableau[i, entering], i))

    if not ratios:
        raise ValueError("The problem is unbounded.")

    _, leaving_row = min(ratios)
    leaving = basis[leaving_row]

    print(f"Entering variable: {variables[entering]}")
    print(f"Leaving variable : {variables[leaving]}")

    # Pivot.
    pivot = tableau[leaving_row, entering]
    tableau[leaving_row] /= pivot

    for i in range(len(basis)):
        if i != leaving_row:
            tableau[i] -= tableau[i, entering] * tableau[leaving_row]

    basis[leaving_row] = entering
    iteration += 1

# Final solution.
solution = np.zeros(len(variables))
for i, var_index in enumerate(basis):
    solution[var_index] = tableau[i, -1]

Z = C @ solution

print("\n=== FINAL BIG-M SOLUTION ===")
for i, v in enumerate(variables):
    print(f"{v} = {solution[i]:.3f}")

print(f"Optimal objective value Z = {Z:.3f}")

if solution[4] > 1e-7:
    print("Artificial variable is positive, so the original problem is infeasible.")
else:
    print("Artificial variable = 0, so the solution is feasible and optimal.")

import numpy as np

# Balanced transportation problem
cost = np.array([
    [11, 6, 15, 13],
    [8,  8, 13, 18],
    [9,  2, 1,  20]
], dtype=float)

supply = [27, 49, 34]
demand = [19, 15, 23, 53]


def vogel_approximation(cost, supply, demand):
    cost = cost.copy()
    supply = supply.copy()
    demand = demand.copy()

    m, n = cost.shape
    allocation = np.zeros((m, n))
    active_rows = [True] * m
    active_cols = [True] * n

    while any(active_rows) and any(active_cols):
        candidates = []

        # Row penalties
        for i in range(m):
            if active_rows[i]:
                values = [cost[i, j] for j in range(n) if active_cols[j]]
                values.sort()
                penalty = values[1] - values[0] if len(values) > 1 else values[0]
                candidates.append((penalty, 0, i))

        # Column penalties
        for j in range(n):
            if active_cols[j]:
                values = [cost[i, j] for i in range(m) if active_rows[i]]
                values.sort()
                penalty = values[1] - values[0] if len(values) > 1 else values[0]
                candidates.append((penalty, 1, j))

        # Highest penalty; tie -> lowest cost, then largest allocation
        max_penalty = max(x[0] for x in candidates)
        best = None

        for penalty, kind, idx in candidates:
            if penalty != max_penalty:
                continue

            if kind == 0:  # row
                j = min(
                    (j for j in range(n) if active_cols[j]),
                    key=lambda j: cost[idx, j]
                )
                low_cost = cost[idx, j]
                quantity = min(supply[idx], demand[j])
            else:  # column
                i = min(
                    (i for i in range(m) if active_rows[i]),
                    key=lambda i: cost[i, idx]
                )
                low_cost = cost[i, idx]
                quantity = min(supply[i], demand[idx])

            key = (low_cost, -quantity, kind, idx)
            if best is None or key < best[0]:
                best = (key, kind, idx)

        _, kind, idx = best

        if kind == 0:
            i = idx
            j = min(
                (j for j in range(n) if active_cols[j]),
                key=lambda j: cost[i, j]
            )
        else:
            j = idx
            i = min(
                (i for i in range(m) if active_rows[i]),
                key=lambda i: cost[i, j]
            )

        quantity = min(supply[i], demand[j])
        allocation[i, j] = quantity

        supply[i] -= quantity
        demand[j] -= quantity

        if supply[i] == 0 and demand[j] == 0:
            # For this assignment's data, degeneracy does not occur.
            active_rows[i] = False
            active_cols[j] = False
        elif supply[i] == 0:
            active_rows[i] = False
        elif demand[j] == 0:
            active_cols[j] = False

    return allocation


allocation = vogel_approximation(cost, supply, demand)
total_cost = np.sum(cost * allocation)

print("=== VAM: Vogel's Approximation Method ===")
print("\nCost Matrix:")
print(cost.astype(int))

print("\nSupply:", supply)
print("Demand:", demand)

print("\nInitial Basic Feasible Solution (VAM Allocation):")
print(allocation.astype(int))

print("\nInitial Transportation Cost =", int(total_cost))

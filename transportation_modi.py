import numpy as np

# Same transportation problem used for VAM
cost = np.array([
    [11, 6, 15, 13],
    [8,  8, 13, 18],
    [9,  2, 1,  20]
], dtype=float)

# VAM initial allocation
allocation = np.array([
    [0,  4,  0, 23],
    [19, 0,  0, 30],
    [0, 11, 23, 0]
], dtype=float)


def calculate_potentials(cost, allocation):
    m, n = cost.shape
    basic = [(i, j) for i in range(m) for j in range(n)
             if allocation[i, j] > 0]

    u = [None] * m
    v = [None] * n
    u[0] = 0

    changed = True
    while changed:
        changed = False
        for i, j in basic:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i, j] - u[i]
                changed = True
            elif u[i] is None and v[j] is not None:
                u[i] = cost[i, j] - v[j]
                changed = True

    return u, v


def reduced_costs(cost, allocation, u, v):
    m, n = cost.shape
    delta = np.full((m, n), np.nan)

    for i in range(m):
        for j in range(n):
            if allocation[i, j] == 0:
                delta[i, j] = cost[i, j] - (u[i] + v[j])

    return delta


def find_loop(allocation, start):
    m, n = allocation.shape
    basic = {(i, j) for i in range(m) for j in range(n)
             if allocation[i, j] > 0}
    basic.add(start)

    def dfs(path, move_in_row):
        i, j = path[-1]

        if move_in_row:
            candidates = [(i, jj) for jj in range(n)
                          if (i, jj) in basic and (i, jj) != path[-1]]
        else:
            candidates = [(ii, j) for ii in range(m)
                          if (ii, j) in basic and (ii, j) != path[-1]]

        for cell in candidates:
            if cell == start and len(path) >= 4:
                return path + [start]

            if cell in path:
                continue

            result = dfs(path + [cell], not move_in_row)
            if result:
                return result

        return None

    return dfs([start], True) or dfs([start], False)


def modi(cost, allocation):
    allocation = allocation.copy()
    iteration = 0

    while True:
        u, v = calculate_potentials(cost, allocation)
        delta = reduced_costs(cost, allocation, u, v)

        print(f"\n--- MODI Iteration {iteration} ---")
        print("u =", u)
        print("v =", v)
        print("Delta (c - u - v):")
        print(delta)

        negative = np.argwhere(delta < 0)

        if len(negative) == 0:
            print("\nAll Delta >= 0, so the solution is OPTIMAL.")
            return allocation

        # Entering cell = most negative delta
        entering = min(
            [tuple(x) for x in negative],
            key=lambda x: delta[x]
        )

        loop = find_loop(allocation, entering)
        if loop is None:
            raise RuntimeError("Improvement loop could not be found.")

        minus_cells = loop[1:-1:2]
        theta = min(allocation[cell] for cell in minus_cells)

        print("Entering cell =", entering)
        print("Closed loop =", loop)
        print("Theta =", int(theta))

        for k, cell in enumerate(loop[:-1]):
            if k % 2 == 0:
                allocation[cell] += theta
            else:
                allocation[cell] -= theta

        allocation[np.abs(allocation) < 1e-9] = 0
        iteration += 1


final_allocation = modi(cost, allocation)
final_cost = np.sum(cost * final_allocation)

print("\n=== FINAL OPTIMAL SOLUTION ===")
print("Optimal Allocation Matrix:")
print(final_allocation.astype(int))
print("\nMinimum Transportation Cost =", int(final_cost))

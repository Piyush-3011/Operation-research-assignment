# Assignment 1 — Optimization Methods

This repository contains the complete implementation for:
1. Big-M Simplex Method
2. Transportation Problem using VAM
3. Transportation Problem optimization using MODI

## Files
- `big_m.py` — Big-M Simplex implementation
- `big_m_output.txt` — Big-M output
- `transportation_vam.py` — VAM implementation
- `transportation_vam_output.txt` — VAM output
- `transportation_modi.py` — MODI implementation
- `transportation_modi_output.txt` — MODI output
- `Complete_Optimization_Assignment.pdf` — submission report

## Results
### Big-M
Maximize Z = 3x1 + 2x2
subject to x1 + x2 >= 4, 2x1 + x2 <= 5, x1,x2 >= 0.

Optimal solution: x1 = 0, x2 = 5
Optimal objective value: Z = 10

### Transportation Problem
Cost matrix:
S1: 11, 6, 15, 13; supply 27
S2: 8, 8, 13, 18; supply 49
S3: 9, 2, 1, 20; supply 34
Demand: 19, 15, 23, 53

VAM initial cost = 1060.
MODI optimal cost = 1048.

Final allocation:
S1 -> D4 = 27
S2 -> D1 = 19
S2 -> D2 = 4
S2 -> D4 = 26
S3 -> D2 = 11
S3 -> D3 = 23

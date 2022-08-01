"""
This python file is used to analyze if the model is convex
"""


import numpy as np
from sympy import *

x1, x2, w1, w2, b, t = symbols("x1 x2 w1 w2 b t")

# inp = Matrix(1, 2, [u, i])
# w = Matrix(2, 1, [w1, w2])
y = x1*w1 + x2*w2 + b
L = (y-t)**2
print("loss function: ", L)
all_symbols = [w1, w2, b]

first_diff = []
second_diff = []
for symbol in all_symbols:
    first_diff.append(diff(L, symbol))
    temp_diff = []
    for symbol in all_symbols:
        temp_diff.append(diff(first_diff[-1], symbol))
    second_diff.append(temp_diff)

print("\nfirst grad:")
for function in first_diff:
    print(function)
    # print(function.subs([(x1, 0), (x2, 0), (w1, 8.6965e-05), (w2, 6.3001e-05), (b, 3.6654), (t, 5)]))

print("\nHessian: ")
for row in second_diff:
    print(row)
    # for elem in row:
        # print(elem.subs([(x1, 0), (x2, 47), (w1, 8.6965e-05), (w2, 6.3001e-05), (b, 3.6654), (t, 5)]), end=" ")
    print()

Hessian = Matrix(second_diff)
# print(Hessian)
print("\nValue: ", Hessian.det())
from pp_1 import pprint
import numpy as np
from scipy import linalg, stats


print('1:')
matrix = np.array([[1, -4, 2, 1.4],
                   [2, -3.5, 9, 0],
                   [7, 5, -4, 3],
                   [1, 2, 3, 4]])
pprint(matrix)
print('-------------------------------------------------')

print('2:')
P, L, U = linalg.lu(matrix)
print('P:')
pprint(P)
print('L:')
pprint(L)
print('U:')
pprint(U)
print('-------------------------------------------------')

print('3:')
u_d = U.diagonal().prod()
l_d = L.diagonal().prod()
p_inv = np.linalg.det(np.linalg.inv(P))
print("diag U:")
print(u_d)
print("diag L:")
print(l_d)
print("det P^(-1):")
print(p_inv)
print(u_d*l_d*p_inv)
print('-------------------------------------------------')

print("4:")
print("vector 1:")
vector_1 = stats.norm.rvs(0, 100, size=100)
pprint(vector_1)
print("vector 2:")
vector_2 = stats.uniform.rvs(size=100)
pprint(vector_2)
print('-------------------------------------------------')

print('5:')
print('     Vector 1                Vector 2')
print("mean:")
print('\t', np.mean(vector_1), '\t', np.mean(vector_2))
print("mode:")
print('\t', stats.mode(vector_1), '\t', stats.mode(vector_2))
print("median:")
print('\t', np.median(vector_1), '\t', np.median(vector_2))
print("min:")
print('\t', min(vector_1), '\t', min(vector_2))
print("max:")
print('\t', max(vector_1), '\t', max(vector_2))
print("std:")
print('\t', np.std(vector_1), '\t', np.std(vector_2))
print('-------------------------------------------------')

print('6:')
print(stats.chisquare(vector_1))
print(stats.chisquare(vector_2))
print('-------------------------------------------------')

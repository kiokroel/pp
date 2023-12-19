import scipy
import math
import sympy
from pp_1 import pprint


def func(x):
    return sympy.sin(2*x) / sympy.cos(x)


def func_sp():
    x = sympy.Symbol('x')
    return sympy.sin(2 * x) / sympy.cos(x)


def func_opt(x):
    x1 = x[0]
    x2 = x[1]
    return (x1 - 3)**2 + (x2 - 8)**2


def constrain1(x):
    return 3*x[0] + 4*x[1] + 6.0


def constrain2(x):
    return -2*x[0] + x[1] - 2.0


def solution_1():
    return f'{scipy.misc.derivative(func, 2)} \n{scipy.misc.derivative(func, 2, n=2)}'


def solution_2():
    x = sympy.Symbol('x')
    return f'{sympy.diff(func_sp(), x)} \n{sympy.diff(func_sp(), x, 3)}'


def solution_3():
    return scipy.integrate.quad(func, 2, 3)[0]


def solution_4():
    return sympy.integrate(func_sp())


def solution_5():
    x0 = [0, 5]
    b = (0.0, math.inf)
    bnds = (b, b)
    con1 = {'type': 'eq', 'fun': constrain1}
    con2 = {'type': 'ineq', 'fun': constrain2}
    cons = [con1, con2]
    solution = scipy.optimize.minimize(func_opt, x0, method='SLSQP',
                                       bounds=bnds, constraints=cons)
    return solution


if __name__ == '__main__':
    print(solution_1())
    print("-----------------------------------------------")
    print(solution_2())
    print("-----------------------------------------------")
    print(solution_3())
    print("-----------------------------------------------")
    print(solution_4())
    print("-----------------------------------------------")
    print(solution_5())
    print("-----------------------------------------------")



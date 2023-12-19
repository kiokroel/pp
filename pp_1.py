import numpy as np


def solution_1():
    my_array = np.arange(10, 70, 2)
    return my_array


def solution_2():
    A = np.array(solution_1().reshape((6, 5)))
    A = np.transpose(A)
    return A


def solution_3():
    A = solution_2()
    A = A * 2.5
    A[0] = A[0] - 5
    return A


def solution_4():
   B = np.random.randint(0, 11, (6, 3))
   return B


def solution_5():
    A = solution_3()
    B = solution_4()
    a = np.sum(A, axis=1)
    b = np.sum(B, axis=0)
    return a, b, len(a), len(b)


def solution_6():
    A = solution_3()
    B = solution_4()
    return np.dot(A, B)


def solution_7():
    A = solution_3()
    A = np.delete(A, 2, 1)
    B = solution_4()
    B = np.concatenate((B, np.random.randint(10, 21, (len(B), 3))), axis=1)
    return A, B


def solution_8():
    A, B = solution_7()
    det_A = np.linalg.det(A)
    if det_A:
        A_inv = np.linalg.inv(A)
        pprint(A_inv)
    det_B = np.linalg.det(B)
    if det_B:
        B_inv = np.linalg.inv(B)
        pprint(B_inv)


def solution_9():
    A, B = solution_7()
    np.linalg.matrix_power(A, 6)
    np.linalg.matrix_power(B, 14)
    return np.linalg.matrix_power(A, 6), np.linalg.matrix_power(B, 14)


def solution_10():
    sole = np.array([[1, -4, 2, 1.4], [2, -3.5, 9, 0], [7, 5, -4, 3], [1, 2, 3, 4]])
    b = np.array([20, 7.8, -6, 6])
    roots = np.linalg.solve(sole, b)
    return roots


def pprint(array, separator='|', max_num_len=6, len_befor_dot=3):
    ret_str = separator
    for i in range(len(array)):
        if type(array[i]) == np.ndarray:
            pprint(array[i])
        else:
            if (array[i] % 1):
                ret_str += ("{:" + f'{max_num_len}.{len_befor_dot}' + "f}").format(array[i]) + separator
            else:
                if abs(array[i]) > 10**(max_num_len-1):
                    num_len = len(str(array[i]))
                    if str(array[i])[0] == "-":
                        new_num = int(str(array[i])[0:max_num_len + 1])
                    else:
                        new_num = int(str(array[i])[0:max_num_len])
                    num = ("{:" + f'{max_num_len}' + "d}").format(new_num)
                    ret_str += f'{num}*10^({num_len-max_num_len})' + separator
                else:
                    if int(array[i]) < 0:
                        ret_str += ("{:" + f'{max_num_len}d' + "}").format(int(array[i])) + separator
                    else:
                        ret_str += ("{:" + f'{max_num_len}d' + "}").format(int(array[i])) + separator
    if (ret_str != separator):
        print(ret_str)


if __name__ == '__main__':
    pprint(solution_1())
    print("-----------------------------------------------")
    pprint(solution_2())
    print("-----------------------------------------------")
    pprint(solution_3())
    print("-----------------------------------------------")
    pprint(solution_4())
    print("-----------------------------------------------")
    a, b, a_len, b_len = solution_5()
    pprint(a)
    print(a_len)
    print()
    pprint(b)
    print(b_len)
    print("-----------------------------------------------")
    pprint(solution_6())
    print("-----------------------------------------------")
    a, b = solution_7()
    pprint(a)
    print()
    pprint(b)
    print("-----------------------------------------------")
    solution_8()
    print("-----------------------------------------------")
    a, b = solution_9()
    pprint(a)
    print()
    pprint(b)
    print("-----------------------------------------------")
    pprint(solution_10())

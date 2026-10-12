import numpy as np
import math
import cmath
from broadcast import broadcast_arrays


# element-wise unary functions
def negative(a):
    if isinstance(a, list):
        return [negative(ai) for ai in a]

    return -a

def exp(a):
    if isinstance(a, list):
        return [exp(ai) for ai in a]

    if isinstance(a, complex):
        return cmath.exp(a)

    return math.exp(a)

def log(a):
    if isinstance(a, list):
        return [log(ai) for ai in a]

    elif a == 0:
        return float('-inf')

    elif a < 0:
        return float('nan')

    return math.log(a)

# element-wise binary functions
def elementwise_binary(a, b, operation):
    if isinstance(a, list):
        return [elementwise_binary(ai, bi, operation) for ai, bi in zip(a, b)]

    return operation(a, b)

def power(x1, x2, dtype = None):

    def power_element(x, y):
        result = x ** y

        if isinstance(result, complex) and dtype is None:
            return float("nan")

        return result

    result = elementwise_binary(x1, x2, power_element)

    if dtype is not None:
        result = conv_dtype(result, dtype)

    return result

def add(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x + y)

def subtract(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x - y)

def multiply(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x * y)

def true_divide(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x / y)

def greater(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x > y)

def less(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x < y)

def conv_dtype(a, dtype):
    if isinstance(a, list):
        return [conv_dtype(ai, dtype) for ai in a]

    return dtype(a)

def sum(a, axis=None, dtype=None, where=None, initial=None):
    if dtype is not None:
        a = conv_dtype(a, dtype)

    if a == []:
        return 0 if initial is None else initial
        
    if where is not None:
        if axis == 1:
            result = []

            for row in a:
                selected = []
                for x, keep in zip(row, where):
                    if keep:
                        selected.append(x)

                result.append(sum(selected))
            return result
    
    if axis is None:
        if not isinstance(a, list):
            return a

        total = 0 if initial is None else initial

        for ai in a:
            total += sum(ai)

        return total

    elif axis == 0:
        return [sum([row[i] for row in a], initial = initial) for i in range(len(a[0]))]

    elif axis == 1:
        return [sum(row, initial = initial) for row in a]

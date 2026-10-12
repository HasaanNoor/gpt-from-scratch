import numpy as np

def shape(a):
    result = []

    while isinstance(a, list):
        result.append(len(a))

        if len(a) == 0:
            break

        a = a[0]

    return tuple(result)


def multiply_scalar(scalar, a):
    if not isinstance(a, list):
        return scalar * a

    return [multiply_scalar(scalar, ai) for ai in a]


def vector_dot(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length!")

    total = 0
    for ai, bi in zip(a, b):
        total += ai * bi

    return total


def vector_matrix_dot(a, b):
    if len(a) != len(b):
        raise ValueError("Vector and matrix cannot be multiplied, incompatible dimensions!")

    result = []

    for j in range(len(b[0])):
        column = [row[j] for row in b]
        result.append(vector_dot(a, column))

    return result


def dot(a, b):
    shape_a = shape(a)
    shape_b = shape(b)

    ndim_a = len(shape_a)
    ndim_b = len(shape_b)
 
    # Scalar involved
    if ndim_a == 0:
        return multiply_scalar(a, b)

    elif ndim_b == 0:
        return multiply_scalar(b, a)

    # 1D x 1D
    elif ndim_a == 1 and ndim_b == 1:
        return vector_dot(a, b)

    # strip away layers of a until it becomes 1D
    elif ndim_a > 1:
        return [dot(sub_list, b) for sub_list in a]

    # strip away layers of b until it becomes 2D
    elif ndim_b > 2:
        return [dot(a, sub_list) for sub_list in b]

    # 1D x 2D
    else:
        return vector_matrix_dot(a, b)
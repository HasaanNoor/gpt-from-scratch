import numpy as np

def shape(a):
    if not isinstance(a, list):
        return ()

    if len(a) == 0:
        return (0,)

    first_shape = shape(a[0])

    for item in a[1:]:
        if shape(item) != first_shape:
            raise ValueError("All dimensions must be consistent")

    return (len(a), ) + first_shape

# only for 2D x 2D matrix case
def multiply_check(a, b):
    rows_a, cols_a = shape(a)
    rows_b, cols_b = shape(b)

    return cols_a == rows_b

def matrix_multiply(a, b):

    shape_a = shape(a)
    shape_b = shape(b)

    a_is_vector = not isinstance(a[0], list)
    b_is_vector = not isinstance(b[0], list)

    # 1D x 1D
    if a_is_vector and b_is_vector:
        return vector_dot(a, b)

    # 1D x ND
    if a_is_vector and len(shape_b) > 2:
        return vector_batch_mult(a, b)

    # ND x 1D
    if len(shape_a) > 2 and b_is_vector:
        return batch_vector_mult(a, b)

    # 1D x 2D
    if a_is_vector and not b_is_vector:
        return vector_matrix_mult(a, b)

    # 2D x 1D
    if not a_is_vector and b_is_vector:
        return matrix_vector_mult(a, b)

    # 2D x 2D
    if len(shape_a) == 2 and len(shape_b) == 2:

        if not multiply_check(a, b):
            raise ValueError("Matrices cannot be multiplied")

        rows_a, cols_a = shape(a)
        rows_b, cols_b = shape(b)

        result = []

        for i in range(rows_a):
            row = []

            for j in range(cols_b):
                total = 0
                for k in range(len(a[0])):
                    total += a[i][k] * b[k][j]

                row.append(total)

            result.append(row)

        return result

    # ND x ND
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]

    matrix_a = shape_a[-2:]
    matrix_b = shape_b[-2:]

    if matrix_a[1] != matrix_b[0]:
        raise ValueError("Matrices cannot be multiplied")

    return batch_broadcast_mult(a, b, batch_a, batch_b)

# 1D x 1D
def vector_dot(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length")

    total = 0

    for ai, bi in zip(a, b):
        total += ai * bi

    return total

# 2D x 1D
def matrix_vector_mult(a, b):
    result = []

    for row in a:
        result.append(vector_dot(row, b))

    return result

# 1D x 2D
def vector_matrix_mult(a, b):
    result = []

    for j in range(len(b[0])):
        col = []
        for i in range(len(b)):
            col.append(b[i][j])
        result.append(vector_dot(a, col))

    return result

# 1D x ND
def vector_batch_mult(a, b):

    if len(shape(b)) == 2:
        return vector_matrix_mult(a, b)

    result = []

    for sub_batch in b:
        result.append(vector_batch_mult(a, sub_batch))

    return result

# ND x 1D
def batch_vector_mult(a, b):

    if len(shape(a)) == 2:
        return matrix_vector_mult(a, b)

    result = []

    for sub_batch in a:
        result.append(batch_vector_mult(sub_batch, b))

    return result


# which matrix to use from a particular batch shape?
def broadcast_index(batch_shape, result_index):
    index = []

    offset = len(result_index) - len(batch_shape)

    for i in range(len(batch_shape)):
        if batch_shape[i] == 1:
            index.append(0)
        else:
            index.append(result_index[i + offset])

    return tuple(index)

# for finding out which matrix we need from a particular batch
def get_matrix(a, index):
    current = a

    for i in index:
        current = current[i]

    return current

# takes batch shape and comes up with every possible index tuple
def all_indices(shape):
    if len(shape) == 0:
        return [()]

    result = []
    first_dim = shape[0]
    rest = shape[1:]
    rest_indices = all_indices(rest)

    for i in range(first_dim):
        for index in rest_indices:
            result.append((i,) + index)

    return result
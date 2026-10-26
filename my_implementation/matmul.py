from broadcast import *

def matrix_2d(a, b):
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])

    if cols_a != rows_b:
        raise ValueError("Matrices cannot be multiplied")

    result = []

    for i in range(rows_a):
        row = []

        for j in range(cols_b):
            total = 0

            for k in range(cols_a):
                total += a[i][k] * b[k][j]
            row.append(total)

        result.append(row)
    return result


def batch_matmul(a, b):
    if len(shape(a)) == 2:
        return matrix_2d(a, b)

    result = []

    for sub_a, sub_b in zip(a, b):
        result.append(batch_matmul(sub_a, sub_b))

    return result


def matmul(a, b):
    shape_a = shape(a)
    shape_b = shape(b)

    if len(shape_a) == 0 or len(shape_b) == 0:
        raise ValueError("Matrix multiplication does not support scalar operands")

    a_vector = len(shape_a) == 1
    b_vector = len(shape_b) == 1

    # Convert vectors into 2D matrices
    if a_vector:
        a = [a]

    if b_vector:
        b = [[x] for x in b]

    shape_a = shape(a)
    shape_b = shape(b)

    # Check matrix multiplication compatibility
    if shape_a[-1] != shape_b[-2]:
        raise ValueError("Matrices cannot be multiplied")

    # Determine common batch shape
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]

    common_batch = broadcast_shapes(batch_a, batch_b)

    # Broadcast batch dimensions, preserving matrix dimensions
    a = broadcast_to(a, common_batch + shape_a[-2:])
    b = broadcast_to(b, common_batch + shape_b[-2:])

    # Perform multiplication
    result = batch_matmul(a, b)

    # Remove temporary dimensions introduced for vectors
    if a_vector:
        if b_vector:
            return result[0][0]

        def remove_first_matrix_dim(x, depth):
            if depth == 0:
                return x[0]
            return [remove_first_matrix_dim(item, depth - 1) for item in x]

        result = remove_first_matrix_dim(result, len(common_batch))

    if b_vector:
        def remove_last_matrix_dim(x, depth):
            if depth == 0:
                return [row[0] for row in x]
            return [remove_last_matrix_dim(item, depth - 1) for item in x]

        result = remove_last_matrix_dim(result, len(common_batch))

    return result
import math
from from_scratch import from_scratch

@from_scratch('numpy.shape')
def shape(a):
    result = []

    while isinstance(a, list):
        result.append(len(a))

        if len(a) == 0:
            break

        a = a[0]

    return tuple(result)


@from_scratch('numpy.broadcast_shapes')
def broadcast_shapes(*shapes):
    reversed_shapes = [shape[::-1] for shape in shapes]
    ndims = [len(shape) for shape in reversed_shapes]
    max_dims = max(ndims)

    result = []

    for i in range(max_dims):
        dims = []
        for shape in reversed_shapes:
            if i < len(shape):
                dims.append(shape[i])

        non_ones = []
        for dim in dims:
            if dim != 1:
                non_ones.append(dim)

        if len(non_ones) > 0:
            first_dim = non_ones[0]

            for dim in non_ones:
                if dim != first_dim:
                    raise ValueError("Shapes cannot be broadcast together")

            result.append(first_dim)

        else:
            result.append(1)

    result.reverse()
    return tuple(result)

@from_scratch('numpy.broadcast_to')
def broadcast_to(array, target_shape):

    array_shape = shape(array)

    if len(array_shape) > len(target_shape):
        raise ValueError("Array cannot be broadcast to the target shape")

    num_missing_dims = len(target_shape) - len(array_shape)
    padded_shape = (1,) * num_missing_dims + array_shape

    for original_dim, target_dim in zip(padded_shape, target_shape):
        if original_dim != target_dim and original_dim != 1:
            raise ValueError("Array cannot be broadcast to the target shape")

    padded_array = array

    for _ in range(num_missing_dims):
        padded_array = [padded_array]

    def broadcast(padded_array, current_shape, target_shape):
        if len(target_shape) == 0:
            return padded_array
        
        current_dim = current_shape[0]
        target_dim = target_shape[0]

        if current_dim == target_dim:
            return [broadcast(padded_array[i], current_shape[1:], target_shape[1:]) for i in range(target_dim)]
        
        return [broadcast(padded_array[0], current_shape[1:], target_shape[1:]) for _ in range(target_dim)]
    
    return broadcast(padded_array, padded_shape, target_shape)

@from_scratch('numpy.broadcast_arrays')
def broadcast_arrays(*arrays):

    shapes = []

    for array in arrays:
        shapes.append(shape(array))

    common_shape = broadcast_shapes(*shapes)
    result = []

    for array in arrays:
        result.append(broadcast_to(array, common_shape))

    return tuple(result)

# element-wise unary functions
@from_scratch('numpy.negative')
def negative(a):
    if isinstance(a, list):
        return [negative(ai) for ai in a]

    return -a

@from_scratch('numpy.exp')
def exp(a):
    if isinstance(a, list):
        return [exp(ai) for ai in a]

    if isinstance(a, complex):
        real = math.exp(a.real)
        return real * complex(math.cos(a.imag), math.sin(a.imag))

    return math.exp(a)

@from_scratch('numpy.log')
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

@from_scratch('numpy.power')
def power(x1, x2, dtype = None):
    x1, x2 = broadcast_arrays(x1, x2)
    def power_element(x, y):
        result = x ** y

        if isinstance(result, complex) and dtype is None:
            return float("nan")

        return result

    result = elementwise_binary(x1, x2, power_element)

    if dtype is not None:
        result = conv_dtype(result, dtype)

    return result

@from_scratch('numpy.add')
def add(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x + y)

@from_scratch('numpy.subtract')
def subtract(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x - y)

@from_scratch('numpy.multiply')
def multiply(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x * y)

@from_scratch('numpy.true_divide')
def true_divide(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x / y)

@from_scratch('numpy.greater')
def greater(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x > y)

@from_scratch('numpy.less')
def less(a, b):
    a, b = broadcast_arrays(a, b)
    return elementwise_binary(a, b, lambda x, y: x < y)

def conv_dtype(a, dtype):
    if isinstance(a, list):
        return [conv_dtype(ai, dtype) for ai in a]

    return dtype(a)

@from_scratch('numpy.sum')
def sum(a, axis=None, dtype=None, where=None, initial=None):
    if dtype is not None:
        a = conv_dtype(a, dtype)

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

@from_scratch('numpy.dot')
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
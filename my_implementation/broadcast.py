import numpy as np

def shape(a):
    result = []

    while isinstance(a, list):
        result.append(len(a))

        if len(a) == 0:
            break

        a = a[0]

    return tuple(result)

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


def broadcast_arrays(*arrays):

    shapes = []

    for array in arrays:
        shapes.append(shape(array))

    common_shape = broadcast_shapes(*shapes)
    result = []

    for array in arrays:
        result.append(broadcast_to(array, common_shape))

    return result
def shape(a):
    result = []

    while isinstance(a, list):
        result.append(len(a))

        if len(a) == 0:
            break

        a = a[0]

    return tuple(result)


    
def transpose(a, axes=None):
    a_shape = shape(a)
    ndims = len(a_shape)

    if axes is None:
        axes = tuple(range(ndims))[::-1]

    if sorted(axes) != list(range(ndims)):
        raise ValueError("Invalid axes")

    if ndims <= 1:
        return a

    elif ndims == 2:
        rows, cols = a_shape

        if axes == (0, 1):
            return a

        return [[a[i][j] for i in range(rows)] for j in range(cols)]

    else:
        new_shape = tuple(a_shape[i] for i in axes)
        def build(indices):
            if len(indices) == ndims:
                original_indices = tuple(indices[axes.index(i)] for i in range(ndims))

                value = a

                for i in original_indices:
                    value = value[i]

                return value

            return [build(indices + (i,)) for i in range(new_shape[len(indices)])]
        return build(())



    
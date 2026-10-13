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
    ndims = len(shape(a_shape))

    if ndims <= 1:
        return a

    elif ndims == 2:
        rows, cols = a_shape
        return [[a[i][j] for i in range(rows)] for j in range(cols)]

    else:
        rev_shape = a_shape[::-1]

        def build(indices):
            if len(indices) == ndims:
                value = a

                for i in indices[::-1]:
                    value = value[i]

                return value
            return [build(indices + (i,)) for i in range(rev_shape[len(indices)])]

        return build(())




    
import math
import random
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

@from_scratch('numpy.random.rand')
def rand(*shape):

    if not shape:
        return random.random()

    return [rand(*shape[1:]) for _ in range(shape[0])]

@from_scratch('numpy.zeros')
def zeros(shape, dtype=float):
    if isinstance(shape, int):
        shape = (shape,)

    def zero_value():
        if isinstance(dtype, list):
            return tuple(0 for _ in dtype)
        
        return dtype(0)

    def build(dim):
        if dim == len(shape):
            return zero_value()
        return [build(dim + 1) for _ in range(shape[dim])]

    return build(0)

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

@from_scratch('numpy.matmul')
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

@from_scratch('numpy.transpose')
def transpose(a, axes=None):
    a_shape = shape(a)
    ndims = len(a_shape)
    if axes is None:
        axes = tuple(range(ndims))[::-1]
    else:
        if len(axes) != ndims:
            raise ValueError("Invalid axes")

        if any(axis < -ndims or axis >= ndims for axis in axes):
            raise ValueError("Invalid axes")

        axes = tuple(axis % ndims for axis in axes)

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

class MyModule:
    model_eval = False

    def register_buffer(self, name, data):
        setattr(self, name, data)

    def eval(self):
        MyModule.model_eval = True
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Module [Module] is missing the required "forward" function')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, d_out):
        raise NotImplementedError()

    def parameters(self):
        return []

    def reset_parameters(self):
        return

    def zero_grad(self, set_to_none=True):
        for param in self.parameters():
            param.zero_grad(set_to_none=set_to_none)

    def get_parameter(self, name):
        return getattr(self, name)

    def __repr__(self):
        return self.__target_path__.split('.')[-1] + "()"

@from_scratch('torch.nn.Parameter')
class MyParameter:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
    def __getattr__(self, name):
        if name == 'grad':
            return self.grad
        if name == 'T':
            pass
        return self.data.__getattr__(name)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        s = 'Parameter containing:\n'
        s += repr(self.data)
        return s

    def zero_grad(self, set_to_none=True):
        if self.requires_grad:
            if set_to_none:
                self.grad = None
            elif self.grad is not None:
                self.grad[:] = 0

    def backward(self, d_out):
        if self.grad is None:
            self.grad = d_out
        else:
            self.grad[:] = d_out

@from_scratch('torch.nn.Sigmoid')
class MySigmoid(MyModule):

    def forward(self, x):
        from numpy import exp, true_divide
        self.output = true_divide(1, 1 + exp(-x))
        return self.output

    def backward(self, d_out):
        return d_out * self.output * (1 - self.output)

@from_scratch('torch.nn.Linear')
class MyLinear(MyModule):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = None
        self.bias = None

        self.reset_parameters()

    def __repr__(self):
        return (f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})")

    def reset_parameters(self):
        from numpy.random import rand
        from numpy import zeros

        bound = 1 / (self.in_features ** 0.5)
        weights = zeros((self.out_features, self.in_features))
        random_weights = rand(self.out_features, self.in_features)

        for i in range(self.out_features):
            for j in range(self.in_features):
                weights[i][j] = (2 * random_weights[i][j] - 1) * bound

        self.weight = MyParameter(weights)

        if self.use_bias:
            biases = zeros(self.out_features)
            random_biases = rand(self.out_features)

            for i in range(self.out_features):
                biases[i] = (2 * random_biases[i] - 1) * bound

            self.bias = MyParameter(biases)
        else:
            self.bias = None

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def forward(self, x):
        from numpy import matmul, transpose, add

        self.input = x
        output = matmul(x, transpose(self.weight.data))

        if self.bias is not None:
            output = add(output, self.bias.data)
        
        return output

    def backward(self, d_out):
        from numpy import matmul, transpose, sum

        d_input = matmul(d_out, self.weight.data)
        d_weight = matmul(transpose(d_out), self.input)

        self.weight.backward(d_weight)

        if self.bias is not None:
            d_bias = sum(d_out, axis = 0)
            self.bias.backward(d_bias)

        return d_input


@from_scratch('torch.nn.Sequential')
class MySequential(MyModule):

    def __init__(self, *modules):
        self.modules = modules

    def __repr__(self):
        if not self.modules:
            return "Sequential()"

        result = "Sequential(\n"

        for i, module in enumerate(self.modules):
            result += f"  ({i}): {repr(module)}\n"

        result += ")"
        return result

    def forward(self, x):
        for module in self.modules:
            x = module(x)

        return x

    def backward(self, d_out):
        for module in reversed(self.modules):
            d_out = module.backward(d_out)

        return d_out

    def parameters(self):
        params = []

        for module in self.modules:
            params.extend(module.parameters())

        return params



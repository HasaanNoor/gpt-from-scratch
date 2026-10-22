import random

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

def rand(shape):

    if not shape:
        return random.random()

    return [rand(shape[1:]) for _ in range(shape[0])]

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

class MySigmoid(MyModule):

    def forward(self, x):
        from numpy import exp, true_divide
        self.output = true_divide(1, 1 + exp(-x))
        return self.output

    def backward(self, d_out):
        return d_out * self.output * (1 - self.output)
    
class MyLinear(MyModule):

    def __init__(self, in_features, out_features, bias=True):
        import numpy as np

        self.in_features = in_features
        self.out_features = out_features

        bound = 1 / np.sqrt(in_features)

        weights = np.random.uniform(-bound, bound, (out_features, in_features))

        self.weight = MyParameter(weights)

        if bias:
            biases = np.random.uniform(-bound, bound, out_features)
            self.bias = myParameter(biases)

        else:
            self.bias = None

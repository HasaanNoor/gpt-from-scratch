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

def rand(*shape):

    if not shape:
        return random.random()

    return [rand(*shape[1:]) for _ in range(shape[0])]

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

class MyLoss:
    def __init__(self):
        self.value = None
        self.d_in = None
        self.model = None

    def set_model(self, model):
        self.model = model
        
    def item(self):
        return self.value
    
    def forward(self, yhat, y):
        raise NotImplementedError()
    
    def backward(self):
        d_in = self.d_in
        if self.model is not None:
            d_in = self.model.backward(d_in)
        return d_in
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class MyBCELoss(MyLoss):
    def forward(self, yhat, y):
        from numpy import log, sum

        n = 1
        for dim in yhat.shape:
            n *= dim

        loss = -(y * log(yhat) + (1 - y) * log(1 - yhat))
        self.value = sum(loss) / n
        self.d_in = ((1 - y) / (1 - yhat) - y / yhat) / n

        return self

class MyOptimizer:
    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = {}
        self.defaults = defaults
        self.params = params

    def zero_grad(self, set_to_none=True):
        for param in self.params:
            param.zero_grad(set_to_none=set_to_none)

    def step(self):
        raise NotImplementedError()

    def __repr__(self):
        s = f'{self.__target_path__.split('.')[-1]} (\n'
        s += 'Parameter Group 0\n'
        for k in sorted(self.defaults.keys()):
            s += f'    {k}: {self.defaults[k]}\n'
        s += ')'
        return s

class MySGD(MyOptimizer):
    def __init__(self, params, lr=0.001):
        defaults = {
            "lr": lr,
            "momentum": 0,
            "dampening": 0,
            "weight_decay": 0,
            "nesterov": False,
            "maximize": False,
            "foreach": None,
            "differentiable": False,
            "fused": None
        }

        super().__init__(params, defaults)

        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            param.data = param.data - self.lr * param.grad

class MyLogisticRegression(MyModule):
    def __init__(self, num_features):
        self.linear = MyLinear(num_features, 1)
        self.sigmoid = MySigmoid()

    def forward(self, x):
        z = self.linear(x)
        return self.sigmoid(z)

    def backward(self, d_out):
        d_out = self.sigmoid.backward(d_out)
        return self.linear.backward(d_out)

    def parameters(self):
        return self.linear.parameters()

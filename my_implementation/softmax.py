def myMax

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

class MySoftmax(MyModule):
    def __init__(self, dim=None):
        self.dim = dim
        self.output = None

    def __repr__(self):
        return f"Softmax(dim={self.dim})"

    def forward(self, x):
        from numpy import exp, max, sum
        dim = self.dim

        if dim is None:
            if x.ndim in (0, 1, 3):
                dim = 0
            else:
                dim = 1
        self.actual_dim = dim

        shifted  = x - max(x, axis=dim, keepdims=True)
        exponentials = exp(shifted)
        denominator = sum(exponentials, axis=dim, keepdims=True)

        self.output = exponentials / denominator

        return self.output

    def backward(self, d_out):
        from numpy import sum
        projection = sum(d_out * self.output, axis = self.actual_dim, keepdims=True)

        return self.output * (d_out - projection)
    
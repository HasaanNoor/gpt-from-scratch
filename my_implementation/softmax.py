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
    
class MySoftmax(MyModule):
    def __init__(self, dim=None):
        self.dim = dim
        self.output = None

    def __repr__(self):
        return f"Softmax(dim={self.dim})"

    def forward(self, x):
        from numpy import exp, sum

        exponentials = exp(x)
        denominator = sum(exponentials, axis=self.dim, keepdims=True)
        self.output = exponentials / denominator

        return self.output

    def backward(self, d_out):
        from numpy import sum

        weighted_grad_sum = sum(d_out * self.output, axis=self.dim, keepdims=True)
        d_input = self.output * (d_out - weighted_grad_sum)
        return d_input

@from_scratch("torch.nn.CrossEntropyLoss")
class MyCrossEntropyLoss(MyLoss):

    def __repr__(self):
        return "CrossEntropyLoss()"

    def forward(self, logits, target):
        from numpy import exp, log, sum

        exponentials = exp(logits)
        probabilities = exponentials / sum(exponentials, axis=1, keepdims=True)
        batch_size = logits.shape[0]
        losses = -target * log(probabilities)
        self.value = sum(losses) / batch_size
        self.d_in = (probabilities - target) / batch_size
        
        return self
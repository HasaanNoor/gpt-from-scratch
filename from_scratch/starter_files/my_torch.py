# to be placed in the parent folder.

from from_scratch import from_scratch

@from_scratch('torch.nn.Module')
class MyModule:
    def __init__(self):
        self.training = True
        self._modules = {}

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
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
        # do nothing if the module doesn't have any parameters
        return

    def zero_grad(self, set_to_none=True):
        for param in self.parameters():
            param.zero_grad(set_to_none=set_to_none)

    def get_parameter(self, name):
        return getattr(self, name)

    def __repr__(self):
        # works as long as you set self._modules to those members that get printed by torch.
        if hasattr(self, '__target_path__'):
            s = self.__target_path__.split('.')[-1]
        else:
            s = self.__class__.__name__

        if len(self._modules) == 0:
            return s + '(' + self.extra_repr() + ')'

        s += '(\n'
        if self.extra_repr():
            s += self.extra_repr() + '\n'
        for name, module in self._modules.items():
            s += f'  ({name}): {repr(module).replace('\n', '\n  ')}\n'
        s += ')'
        return s

    def extra_repr(self):
        # usually a comma separated series of arguments passed to the constructor including defaults
        return ''

    def children(self):
        return []

@from_scratch('torch.nn.Parameter')
class MyParameter:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

    def __getattr__(self, name):
        return getattr(self.data, name)

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

@from_scratch('torch.optim.Optimizer')
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

# The torch super class _Loss doesn't do much.
# Use this as the parent of your Loss functions
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


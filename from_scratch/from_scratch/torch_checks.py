# These are RMP's course materials for CS 5750: Demystifying LLMs.
# Public sharing of course materials is prohibited.

import textwrap

def check_torch_nn_module():
    docstring = textwrap.dedent("""\
    >>> m = nn.Module()
    >>> m
    Module()
    >>> m()
    NotImplementedError: Module [Module] is missing the required "forward" function
    >>> list(m.parameters())
    []
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_parameter():
    docstring = textwrap.dedent("""\
    >>> x = torch.randn(3, 2)
    >>> x
    tensor([[1.1906, 1.8951],
            [0.0841, 0.5333],
            [0.6354, 1.6627]])
    >>> p = nn.Parameter(x, requires_grad=True)
    >>> p
    Parameter containing:
    tensor([[1.1906, 1.8951],
            [0.0841, 0.5333],
            [0.6354, 1.6627]], requires_grad=True)
    >>> p.grad
    >>> d_out = torch.randn(3, 2)
    >>> d_out
    tensor([[ 0.5761, -0.2450],
            [ 1.5785,  0.4615],
            [ 0.2783, -0.7047]])
    >>> p.backward(d_out)
    >>> p.grad
    tensor([[ 0.5761, -0.2450],
            [ 1.5785,  0.4615],
            [ 0.2783, -0.7047]])
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_linear():
    docstring = textwrap.dedent("""\
    >>> m = nn.Linear(2, 3)
    >>> m
    Linear(in_features=2, out_features=3, bias=True)
    >>> m.weight
    Parameter containing:
    tensor([[ 0.2560,  0.4120],
            [-0.2700, -0.2112],
            [ 0.3767,  0.3679]], requires_grad=True)
    >>> m.bias
    Parameter containing:
    tensor([-0.4196,  0.6724, -0.5592], requires_grad=True)
    >>> list(m.parameters())
    [Parameter containing:
    tensor([[ 0.2560,  0.4120],
            [-0.2700, -0.2112],
            [ 0.3767,  0.3679]], requires_grad=True), Parameter containing:
    tensor([-0.4196,  0.6724, -0.5592], requires_grad=True)]
    >>> m.reset_parameters()
    >>> list(m.parameters())
    [Parameter containing:
    tensor([[ 0.3893,  0.6704],
            [ 0.2115, -0.6325],
            [-0.6345, -0.0537]], requires_grad=True), Parameter containing:
    tensor([0.2191, 0.3561, 0.6025], requires_grad=True)]
    >>> input = torch.randn(5, 2, requires_grad=True)
    >>> input
    tensor([[ 0.4699,  0.1371],
            [-1.3922,  1.5115],
            [-0.8499,  2.3769],
            [ 0.4716,  0.4673],
            [-0.2057, -0.6800]], requires_grad=True)
    >>> output = m(input)
    >>> output.size()
    torch.Size([5, 3])
    >>> output
    tensor([[ 0.4939,  0.3688,  0.2970],
            [ 0.6905, -0.8944,  1.4048],
            [ 1.4818, -1.3271,  1.0142],
            [ 0.7160,  0.1603,  0.2782],
            [-0.3168,  0.7427,  0.7696]], grad_fn=<AddmmBackward0>)
    >>> d_out = torch.randn(5, 3)
    >>> total = (output * d_out).sum() 
    >>> if hasattr(m, '__target_path__'):
    ...     d_in = m.backward(d_out)
    ... else:
    ...     total.backward()
    ...     d_in = input.grad
    >>> d_in
    tensor([[-0.0337, -0.0158],
            [-0.0337, -0.0158],
            [-0.0337, -0.0158],
            [-0.0337, -0.0158],
            [-0.0337, -0.0158]])
    >>> [p.grad for p in m.parameters()]
    [tensor([[-1.5063,  3.8129],
            [-1.5063,  3.8129],
            [-1.5063,  3.8129]]), tensor([5., 5., 5.])]
    """)
    dependent_members = ('torch.nn.Parameter',)
    return docstring, dependent_members

def check_torch_nn_bceloss():
    docstring = textwrap.dedent("""\
    >>> loss = nn.BCELoss()
    >>> loss
    BCELoss()
    >>> input = torch.rand(3, 2, requires_grad=True)
    >>> input
    tensor([[0.7145, 0.0648],
            [0.9015, 0.1755],
            [0.3814, 0.0525]], requires_grad=True)
    >>> target = torch.rand(3, 2, requires_grad=False)
    >>> target
    tensor([[0.8809, 0.9159],
            [0.5624, 0.6942],
            [0.2060, 0.5706]])
    >>> output = loss(input, target)
    >>> output.item()
    1.2635689973831177
    >>> if hasattr(loss, 'backward'):
    ...     output.backward()
    ...     d_in = input.grad
    ... else:
    ...     d_in = output.backward()
    >>> d_in
    tensor([[-0.1359, -2.3398],
            [ 0.6362, -0.5973],
            [ 0.1240, -1.7371]])
    """)
    dependent_members = ('torch.nn.Sigmoid',)
    return docstring, dependent_members

def check_torch_nn_sigmoid():
    docstring = textwrap.dedent("""\
    >>> s = nn.Sigmoid()
    >>> s
    Sigmoid()
    >>> input = torch.randn(10, requires_grad=True)
    >>> input
    tensor([-0.3510,  0.6420, -1.6461, -0.3413, -0.9413, -0.1970, -0.4948, -0.7878,
            -0.2715, -0.3278], requires_grad=True)
    >>> output = s(input)
    >>> output
    tensor([0.4131, 0.6552, 0.1616, 0.4155, 0.2806, 0.4509, 0.3788, 0.3126, 0.4325,
            0.4188], grad_fn=<SigmoidBackward0>)
    >>> d_out = torch.randn(10)
    >>> total = (output * d_out).sum()
    >>> if hasattr(s, '__target_path__'):
    ...     d_in = s.backward(d_out)
    ... else:
    ...     total.backward()
    ...     d_in = input.grad
    >>> d_in
    tensor([0.2425, 0.2259, 0.1355, 0.2429, 0.2019, 0.2476, 0.2353, 0.2149, 0.2454,
            0.2434])
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_softmax():
    docstring = textwrap.dedent("""\
    >>> s = nn.Softmax(dim=1)
    >>> s
    Softmax(dim=1)
    >>> n, m = 10, 3
    >>> input = torch.randn(n, m, requires_grad=True)
    >>> input
    tensor([[ 0.9396,  0.7477,  0.8071],
            [ 1.5843,  1.1894,  0.4524],
            [-0.8344, -2.2776,  0.3526],
            [-1.6919, -0.3458,  1.1989],
            [-0.3219, -1.0687, -0.4843],
            [ 0.8712,  2.5422, -0.1822],
            [-0.0403,  0.7924,  1.2324],
            [ 0.6671,  0.5524,  1.1253],
            [ 0.8327, -0.9189, -0.4828],
            [-1.0345, -0.3924, -0.9616]], requires_grad=True)
    >>> target = torch.zeros(n, m)
    >>> target[torch.arange(n), torch.randint(0, m, (n,))] = 1
    >>> target
    tensor([[0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> output = s(input)
    >>> total = (target * torch.log(output)).sum()
    >>> if hasattr(s, '__target_path__'):
    ...     d_in = s.backward(target/output)
    ... else:
    ...     total.backward()
    ...     d_in = input.grad
    >>> d_in
    tensor([[-0.3702,  0.6944, -0.3243],
            [ 0.4990, -0.3375, -0.1615],
            [-0.2216, -0.0523,  0.2739],
            [-0.0438,  0.8318, -0.7881],
            [-0.4303, -0.2039,  0.6342],
            [ 0.8500, -0.7977, -0.0523],
            [-0.1456,  0.6653, -0.5197],
            [ 0.7120, -0.2567, -0.4553],
            [-0.6936,  0.8797, -0.1861],
            [-0.2515, -0.4780,  0.7295]])
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_sequential():
    docstring = textwrap.dedent("""\
    >>> m = nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())
    >>> m
    Sequential(
    (0): Linear(in_features=3, out_features=2, bias=True)
    (1): Sigmoid()
    )
    >>> list(m.parameters())
    [Parameter containing:
    tensor([[-0.2779, -0.3035,  0.2963],
            [ 0.5672,  0.1121,  0.4518]], requires_grad=True), Parameter containing:
    tensor([0.2635, 0.1223], requires_grad=True)]
    >>> [p.grad for p in m.parameters()]
    [None, None]
    >>> input = torch.randn(5, 3, requires_grad=True)
    >>> input
    tensor([[ 0.2844, -0.5286,  0.4060],
            [ 0.2088, -0.3542,  1.6503],
            [ 1.2544,  1.2781, -1.0209],
            [ 2.0239,  1.7078,  1.1153],
            [ 0.8625,  1.8659,  0.6593]], requires_grad=True)
    >>> output = m(input)
    >>> output
    tensor([[0.6143, 0.6005],
            [0.6904, 0.7204],
            [0.3153, 0.6262],
            [0.3807, 0.8771],
            [0.4141, 0.7537]], grad_fn=<SigmoidBackward0>)
    >>> d_out = torch.randn(5, 2)
    >>> total = (output * d_out).sum()
    >>> if hasattr(m, '__target_path__'):
    ...     d_in = m.backward(d_out)
    ... else:
    ...     total.backward()
    ...     d_in = input.grad
    >>> d_in
    tensor([[ 0.0702, -0.0450,  0.1786],
            [ 0.0549, -0.0423,  0.1543],
            [ 0.0728, -0.0393,  0.1697],
            [-0.0044, -0.0595,  0.1185],
            [ 0.0379, -0.0528,  0.1558]])
    >>> [p.grad for p in m.parameters()]
    [tensor([[1.0693, 0.9303, 0.6515],
            [0.7821, 0.6314, 0.4334]]), tensor([1.1450, 0.9688])]
    """)
    dependent_members = ('torch.nn.Linear', 'torch.nn.Sigmoid')
    return docstring, dependent_members

def check_torch_optim_optimizer():
    docstring = textwrap.dedent("""\
    >>> param = nn.Parameter(torch.randn(3, 2), requires_grad=True)
    >>> params = [param]
    >>> default = dict(key='value')
    >>> optimizer = torch.optim.Optimizer(params, default)
    >>> d_out = torch.randn(3, 2)
    >>> d_out
    tensor([[-1.2229,  1.3587],
            [-0.8582,  0.6064],
            [-0.0857, -0.5980]])
    >>> param.backward(d_out)
    >>> param.grad
    tensor([[-1.2229,  1.3587],
            [-0.8582,  0.6064],
            [-0.0857, -0.5980]])    
    >>> optimizer.zero_grad(set_to_none=False)
    >>> param.grad
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])
    """)
    dependent_members = ('torch.nn.Parameter',)
    return docstring, dependent_members

def check_torch_optim_sgd():
    docstring = textwrap.dedent("""\
    >>> param = nn.Parameter(torch.randn(5, 3))
    >>> # put a gradient into the parameter
    >>> grad = torch.randn(5, 3)
    >>> if hasattr(param, '__target_path__'):
    ...     param.backward(grad)
    ... else:
    ...    (param * grad).sum().backward()
    >>> params = [param]
    >>> params
    >>> grads = [p.grad for p in params]
    >>> grads
    >>> optimizer = torch.optim.SGD(params, lr=0.1)
    >>> grads = [p.grad for p in params]
    >>> grads
    >>> optimizer.step()
    >>> params
    """)
    dependent_members = ('torch.nn.Parameter',)
    return docstring, dependent_members

def check_torch_nn_crossentropyloss():
    docstring = textwrap.dedent("""\
    >>> loss = nn.CrossEntropyLoss()
    >>> loss
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> input
    >>> target = torch.randn(3, 5).softmax(dim=1)
    >>> target
    >>> output = loss(input, target)
    >>> output.item()
    >>> if hasattr(loss, '__target_path__'):
    ...     d_in = output.backward()
    ... else:
    ...     output.backward()
    ...     d_in = input.grad
    >>> d_in     
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_dropout():
    docstring = textwrap.dedent("""\
    >>> m = nn.Dropout(0.5)
    >>> m
    >>> input = torch.randn(2, 5, 3, requires_grad=True)
    >>> m.eval()
    >>> m(input)
    >>> m.train()
    >>> output = m(input)
    >>> output
    >>> d_out = torch.randn(2, 5, 3)
    >>> if hasattr(m, '__target_path__'):
    ...     d_in = m.backward(d_out)
    ... else:
    ...     (d_out * output).sum().backward()
    ...     d_in = input.grad
    >>> d_in
    """)
    dependent_members = ()
    return docstring, dependent_members

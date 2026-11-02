# These are RMP's course materials for CS 5750: Demystifying LLMs.
# Public sharing of course materials is prohibited.

import textwrap

MODULE_DOCTEST = """\
>>> module = {module}
>>> module
>>> input = {input}
>>> input.requires_grad_()
>>> list(module.parameters())
>>> output = module(input)
>>> d_out = torch.randn(output.size())
>>> if hasattr(module, '__target_path__'):
...     d_in = module.backward(d_out)
... else:
...     (d_out * output).sum().backward()
...     d_in = input.grad
>>> d_in
>>> [p.grad for p in module.parameters()]
"""

LOSS_DOCTEST = """\
>>> loss = {loss}
>>> loss
>>> input = {input}
>>> input.requires_grad_()
>>> target = {target}
>>> output = loss(input, target)
>>> if hasattr(loss, '__target_path__'):
...     d_in = output.backward()
... else:
...     output.backward()
...     d_in = input.grad
>>> d_in
"""

def check_torch_nn_module():
    docstring = textwrap.dedent("""\
    >>> module = nn.Module()
    >>> module
    >>> input = torch.randn(2, 3, 4)
    >>> input.requires_grad_()
    >>> list(module.parameters())
    >>> module(input)
    NotImplementedError
    >>> [p.grad for p in module.parameters()]
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_parameter():
    docstring = textwrap.dedent("""\
    >>> p = nn.Parameter(torch.randn(3, 2), requires_grad=True)
    >>> p.grad
    >>> d_out = torch.randn(3, 2)
    >>> p.backward(d_out)
    >>> p.grad
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_linear():
    docstring = MODULE_DOCTEST.format(module='nn.Linear(4, 5)', input='torch.randn(2, 3, 4)')
    dependent_members = ('torch.nn.Parameter',)
    return docstring, dependent_members

def check_torch_nn_bceloss():
    docstring = LOSS_DOCTEST.format(loss='nn.BCELoss()', input='torch.rand(3, 2)', target='(torch.rand(3, 2) * 1e6).softmax(dim=1)')
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_sigmoid():
    docstring = MODULE_DOCTEST.format(module='nn.Sigmoid()', input='torch.randn(10)')
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_softmax():
    docstring = MODULE_DOCTEST.format(module='nn.Softmax(dim=1)', input='torch.randn(10, 3)')
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_sequential():
    docstring = MODULE_DOCTEST.format(module='nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())', input='torch.randn(2, 5, 3)')
    dependent_members = ('torch.nn.Linear', 'torch.nn.Sigmoid')
    return docstring, dependent_members

def check_torch_optim_optimizer():
    docstring = textwrap.dedent("""\
    >>> param = nn.Parameter(torch.randn(3, 2), requires_grad=True)
    >>> params = [param]
    >>> default = dict(key='value')
    >>> optimizer = torch.optim.Optimizer(params, default)
    >>> d_out = torch.randn(3, 2)
    >>> param.backward(d_out)
    >>> param.grad
    >>> optimizer.zero_grad(set_to_none=False)
    >>> param.grad
    """)
    dependent_members = ('torch.nn.Parameter',)
    return docstring, dependent_members

def check_torch_optim_sgd():
    # TODO: add more tests for different hyperparameters
    docstring = textwrap.dedent("""\
    >>> data = torch.randn(5, 3)
    >>> grad = torch.randn(5, 3)
    >>> param = nn.Parameter(data.clone())
    >>> # put a gradient into the parameter
    >>> if hasattr(param, '__target_path__'):
    ...     param.backward(grad)
    ... else:
    ...    (param * grad).sum().backward()
    >>> params = [param]
    >>> grads = [p.grad for p in params]
    >>> optimizer = torch.optim.SGD(params, lr=0.1, weight_decay=0.01, momentum=0.9, nesterov=True)
    >>> optimizer.step()
    >>> params
    >>> optimizer.step()
    >>> params
    >>> # try dampening
    >>> optimizer = torch.optim.SGD(params, lr=0.1, weight_decay=0.01, momentum=0.9, dampening=0.1)
    >>> optimizer.step()
    >>> params
    >>> optimizer.step()
    >>> params
    """)
    dependent_members = ('torch.nn.Parameter',)
    return docstring, dependent_members

def check_torch_optim_adamw():
    # TODO: add more tests for different hyperparameters
    docstring = textwrap.dedent("""\
    >>> data = torch.randn(5, 3)
    >>> grad = torch.randn(5, 3)
    >>> param = nn.Parameter(data.clone())
    >>> # put a gradient into the parameter
    >>> if hasattr(param, '__target_path__'):
    ...     param.backward(grad)
    ... else:
    ...    (param * grad).sum().backward()
    >>> params = [param]
    >>> grads = [p.grad for p in params]
    >>> optimizer = torch.optim.AdamW(params, lr=0.1, amsgrad=False)
    >>> optimizer.step()
    >>> params
    >>> optimizer.step()
    >>> params
    >>> # try amsgrad
    >>> optimizer = torch.optim.AdamW(params, lr=0.1, amsgrad=True)
    >>> optimizer.step()
    >>> params
    >>> optimizer.step()
    >>> params
    """)
    dependent_members = ('torch.nn.Parameter',)
    return docstring, dependent_members

def check_torch_nn_crossentropyloss():
    docstring = LOSS_DOCTEST.format(loss='nn.CrossEntropyLoss()', input='torch.randn(3, 5)', target='torch.randn(3, 5).softmax(dim=1)')
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_dropout():
    docstring = MODULE_DOCTEST.format(module='nn.Dropout(0.5)', input='torch.randn(2, 5, 3)')
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_embedding():
    docstring = textwrap.dedent("""\
    >>> module = nn.Embedding(6, 3)
    >>> module
    >>> input = torch.randint(0, 6, (2, 3, 4))
    >>> list(module.parameters())
    >>> output = module(input)
    >>> d_out = torch.randn(output.size())
    >>> if hasattr(module, '__target_path__'):
    ...     module.backward(d_out)
    ... else:
    ...     (d_out * output).sum().backward()
    >>> [p.grad for p in module.parameters()]
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_functional_softmax():
    docstring = textwrap.dedent("""\
    >>> input = torch.randn(3, 5)
    >>> F.softmax(input)
    >>> F.softmax(input, dim=1)
    >>> F.softmax(input, dim=0)
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_nn_functional_cross_entropy():
    docstring = textwrap.dedent("""\
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randint(5, (3,), dtype=torch.int64)
    >>> loss = F.cross_entropy(input, target)
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5).softmax(dim=1)
    >>> loss = F.cross_entropy(input, target)
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_torch_utils_data_dataloader():
    docstring = textwrap.dedent("""\
    >>> tokenizer = tiktoken.get_encoding("gpt2")
    >>> txt = '''I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)
    ...
    ... "The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. "Of course it's going to send the value of my picture 'way up; but I don't think of that, Mr. Rickham--the loss to Arrt is all I think of." The word, on Mrs. Thwing's lips, multiplied its _rs_ as though they were reflected in an endless vista of mirrors. And it was not only the Mrs. Thwings who mourned. Had not the exquisite Hermia Croft, at the last Grafton Gallery show, stopped me before Gisburn's "Moon-dancers" to say, with tears in her eyes: "We shall not look upon its like again"?
    ...
    ... '''
    >>> dataset = raschka.gpt.GPTDatasetV1(txt, tokenizer, 8, 8)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=0, generator=torch.default_generator)
    >>> list(iter(dataloader))
    """)
    dependent_members = ()
    return docstring, dependent_members


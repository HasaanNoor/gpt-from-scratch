# These are RMP's course materials for CS 5750: Demystifying LLMs.
# Public sharing of course materials is prohibited.

import ast
import copy
import importlib
import math
import operator
import os
import sys
import textwrap
import traceback
from contextlib import contextmanager
from datetime import datetime

from matplotlib.container import BarContainer
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from random import Random
import random
import pytest

# don't allow students to import anything from from_scratch except from_scratch
import numpy
import torch
from torch.utils.data import DataLoader

import gpt

# I was getting: "RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace _inductor_test"
# I used Gemini to debug. I found that calling optimizer.step() in the tests triggered a just-in-time compilation
# that imports torch._dynamo which in turn attempts to register the _inductor_test namespace. That namespace is already
# registered by pytest, causing the error.
# Gemini suggesting importing the library early to avoid the just-in-time import. it worked!
import torch._dynamo

from unittest.mock import patch
import inspect
import doctest
from functools import wraps

ALLOWED = (
    'builtins', 'math', 'random', 'tiktoken', 'from_scratch.from_scratch',
    'numpy.random.poisson', 'torch.utils.data.Dataset', 'torch.utils.data.DataLoader',
    'gpt.create_dataloader_v1', 'numpy.dtype',
)

TESTS = []  # ['gpt.MultiHeadAttention']

def is_allowed(name):
    tokens = name.split('.')
    for i in range(1, len(tokens)+1):
        if '.'.join(tokens[:i]) in ALLOWED:
            return True
    return False


### 1. PATCHES ###

# 1A. Patches for numpy random to generate random numbers the same as torch.
#
# Store original (not patched versions) of key torch random functions and use them to patch numpy random calls
# so that the numpy implementation of torch random function will match with the torch native random number generator.

class OriginalTorch:
    """
    Store original unpatched versions of torch random functions
    """
    def __init__(self):
        keys = ['normal', 'poisson', 'rand', 'randn', 'manual_seed', 'randint', 'randperm', 'Tensor', 'bernoulli']
        members = dict(inspect.getmembers(torch))
        for key in keys:
            if key in members:
                setattr(self, key, members[key])
            else:
                raise KeyError(f'{key} is not a member of torch.')

class NumpyAsTorch:
    """
    Store implementation of numpy random function using original torch as random number generator.
    """

    original_torch = OriginalTorch()

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        if isinstance(loc, numpy.ndarray):
            loc = NumpyAsTorch.original_torch.Tensor(loc)
        if isinstance(scale, numpy.ndarray):
            scale = NumpyAsTorch.original_torch.Tensor(scale)
        if size is None:
            return NumpyAsTorch.original_torch.normal(mean=loc, std=scale).numpy()
        else:
            return NumpyAsTorch.original_torch.normal(mean=loc, std=scale, size=size).numpy()

    @staticmethod
    def poisson(lam=1.0, size=None):
        if size is not None:
            lam = numpy.broadcast_to(lam, size)
        if isinstance(lam, numpy.ndarray):
            lam = NumpyAsTorch.original_torch.Tensor(lam)
        return NumpyAsTorch.original_torch.poisson(lam).numpy()

    @staticmethod
    def rand(*shape):
        return NumpyAsTorch.original_torch.rand(*shape).numpy()

    @staticmethod
    def randn(*shape):
        return NumpyAsTorch.original_torch.randn(*shape).numpy()

    @staticmethod
    def standard_normal(*shape):
        return NumpyAsTorch.original_torch.randn(*shape).numpy()

    @staticmethod
    def seed(a=None):
        NumpyAsTorch.original_torch.manual_seed(a)

    @staticmethod
    def randint(low, high=None, size=None, dtype=int):
        return NumpyAsTorch.original_torch.randint(low, high, size=size).numpy()

    @staticmethod
    def permutation(x):
        return NumpyAsTorch.original_torch.randperm(x).numpy()

    @staticmethod
    def bernoulli(x):
        return NumpyAsTorch.original_torch.bernoulli(x).numpy()

NUMPY_AS_TORCH = NumpyAsTorch()
def numpy_random_patches():
    """
    Patch numpy random functions with their implementations using torch as the random number generator
    """
    numpy_random_module = get_module('numpy.random')
    patches = []
    for k in dir(NUMPY_AS_TORCH):
        if not k.startswith('_') and hasattr(numpy_random_module, k):
            patches.append(patch.object(numpy_random_module, k, new=wrap_inputs_and_outputs(getattr(NUMPY_AS_TORCH, k), to_array, to_locked)))
    return patches


# 1B. Patches for Random to generate random numbers the same as Numpy.
# Store original (not patched) numpy random functions for use in patched random functions to be used by the
# "from scratch" implementation of numpy random functions.

class OriginalNumpyRandom:
    """
    Store the original numpy.random functions for later.
    """
    def __init__(self):
        keys = ['seed', 'random', 'get_state', 'set_state', 'randn', 'randint', 'choice', 'permutation']
        members = dict(inspect.getmembers(numpy.random))
        for key in keys:
            if key in members:
                setattr(self, key, members[key])
            else:
                raise ValueError(f'{key} not a member of numpy.random.')


class MyRandom(Random):
    """
    Override the random methods to match numpy.
    Don't allow from_scratch implementations of numpy.random.mtrand
    """
    original_numpy_random = OriginalNumpyRandom()

    def random(self):
        return MyRandom.original_numpy_random.random()
    def seed(self, a=None, version=2):
        if a is None:
            a = int(datetime.now().timestamp())
        MyRandom.original_numpy_random.seed(a)
        self.gauss_next = None
    def getstate(self):
        return MyRandom.original_numpy_random.get_state()
    def setstate(self, state):
        return MyRandom.original_numpy_random.set_state(state)
    def gauss(self, mu=0.0, sigma=1.0):
        return MyRandom.original_numpy_random.randn() * sigma + mu
    def randint(self, a, b):
        return MyRandom.original_numpy_random.randint(low=a, high=b+1)
    def randrange(self, start, stop=None, step=1):
        if stop is None:
            stop = start
            start = 0
        if step == 1:
            return MyRandom.original_numpy_random.randint(low=start, high=stop)
        else:
            return MyRandom.original_numpy_random.choice(list(range(start, stop, step)))
    def shuffle(self, x):
        a = [x[i] for i in MyRandom.original_numpy_random.permutation(len(x))]
        for i in range(len(a)):
            x[i] = a[i]

RANDOM_AS_NUMPY = MyRandom()
def random_patches():
    """
    Patch the random module so that it generates random numbers like numpy.random.
    """
    patches = []
    patches.append(patch.object(random, '_inst', new=RANDOM_AS_NUMPY))
    for method in random.__all__:
        # only add methods not classes
        if method == method.lower():
            patches.append(patch.object(random, method, new=getattr(RANDOM_AS_NUMPY, method)))
    return patches


def numpy_dtype_patches():
    """
    Patch dtypes to all be np.uint, np.int_ or np.float_.
    """
    patches = []
    numpy_module = get_module('numpy')
    for dtype in ['uint', 'int', 'float']:
        for bits in [8, 16, 32, 64, 96, 128]:
            bits_dtype = f'{dtype}{bits}'
            if dtype in ('int', 'float'):
                dtype = dtype + '_'
            if hasattr(numpy_module, bits_dtype):
                patches.append(patch.object(numpy_module, bits_dtype, new=getattr(numpy_module, dtype)))
    # ignore dtypes in array construction
    original_array = numpy.array
    def mocked_array(obj, dtype=None, **kwargs):
        if isinstance(dtype, str) and dtype.startswith('timedelta64'):
            dtype = None
        try:
            return original_array(obj, dtype=dtype, **kwargs)
        except ValueError:
            pass
    patches.append(patch.object(numpy_module, 'array', new=mocked_array))
    return patches


def mpl_patches():
    patches = []

    def my_imshow(x, *args, **kwargs):
        return x

    def my_plot(x, y=None):
        if y is None:
            return x
        else:
            return x, y

    def my_show(*, block=None):
        pass

    patches.append(patch('matplotlib.pyplot.imshow', new=my_imshow))
    patches.append(patch('matplotlib.pyplot.plot', new=my_plot))
    patches.append(patch('matplotlib.pyplot.show', new=my_show))
    return patches

def get_doctest_patches(module_name):
    """
    Depending on module under test, patch the correct functions.
    """
    patches = numpy_dtype_patches()
    patches += mpl_patches()
    patches.append(patch('timeit.timeit', new=lambda cmd, globals=None, number=None: 0.0))
    return patches

def get_library_to_student_patches(name):
    """
    Mock the function under test so that it replaces the library implementation
    """
    try:
        _, dependent_members = get_custom_doctests(name)
    except KeyError:
        dependent_members = ()

    assert isinstance(dependent_members, tuple)

    names = [name] + list(dependent_members)

    patches = []
    for name_ in names:
        student_path = get_target(name_)

        # mock library function with student implementation
        student_member = get_member(student_path)
        library_member = get_member(name_)

        if inspect.isclass(inspect.unwrap(library_member)):
            # make sure to wrap inputs/outputs of methods of implemented classes.
            new_member = inspect.unwrap(student_member)
            for inherited_class in inspect.getmro(new_member)[1:-1]:  # first and last are new_member and object
                inherited_path = get_member_path(inherited_class)
                if is_allowed(inherited_path):
                    break
                patches.extend(get_library_to_student_patches(inherited_path))
            for member_name, member in inspect.getmembers(student_member):
                if inspect.isroutine(inspect.unwrap(member)) and member_name in student_member.__dict__:
                    member_path = f'{student_path}.{member_name}'
                    patches.append(patch(member_path, new=member))
        patches.append(patch(name_, new=student_member))
    return patches

import warnings as _warnings
_warnings.filterwarnings("ignore", category=UserWarning)


def is_numeric(obj):
    # https://stackoverflow.com/a/500908
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


UFUNC_LIST = ['numpy.' + name for name, obj in numpy.__dict__.items() if isinstance(obj, numpy.ufunc)]
TORCH_DTYPE_LIST = ['torch.' + name for name, obj in torch.__dict__.items() if isinstance(obj, torch.dtype)]
NUMPY_DTYPE_LIST = ['numpy.' + name for name in numpy.sctypeDict.keys()]

def is_dtype(name):
    return name in TORCH_DTYPE_LIST or name in NUMPY_DTYPE_LIST


def get_dependency_patches(member):
    path = member.__target_path__
    module_name = path.split('.')[0]
    assert module_name in ('numpy', 'torch', 'gpt')
    assert not path.startswith('numpy.random.mtrand')

    # mock the dependencies
    patches = []

    # mock all dependencies so they return transformed output
    dependencies = get_member_dependencies(get_member_path(member))
    if len(dependencies) > 0:
        print(f'Wrapping {module_name} outputs: {dependencies}')
    for dependency in dependencies:
        if is_dtype(dependency) or is_allowed(dependency):
            continue
        dependency_member = get_member(dependency)
        dependency_module_name = get_module_name(dependency_member)
        dependency_module_name = dependency_module_name.split('.')[0]
        if is_allowed(dependency_module_name) or is_student_member(dependency):
            continue

        # module_name is the module for the member under test
        # dependeny_module_name is the module for the dependency
        if dependency_module_name == 'numpy':
            input_wrapper = to_array
        elif dependency_module_name in ('torch', 'gpt'):
            input_wrapper = to_tensor
        else:
            input_wrapper = None

        if module_name == 'numpy':
            output_wrapper = to_lol
        elif module_name in ('torch', 'gpt'):
            output_wrapper = to_locked
        else:
            output_wrapper = None

        patches.append(patch(dependency, new=wrap_inputs_and_outputs(dependency_member, input_wrapper, output_wrapper)))

    return patches

def get_random_patches(member):
    path = member.__target_path__
    path_base = path.split('.')[0]
    if path_base == 'numpy':
        return random_patches()
    elif path_base in ('torch', 'gpt'):
        return numpy_random_patches()
    else:
        return []

def method_under_test_wrapper(member):
    path = member.__target_path__
    path_base = path.split('.')[0]
    if path_base == 'numpy':
        return numpy_transform_inputs_and_outputs
    elif path_base in ('torch', 'gpt'):
        return torch_transform_inputs_and_outputs

def get_user_patches(member):
    patches = get_dependency_patches(member)
    patches += get_random_patches(member)
    return patches

_test_stack = []
@contextmanager
def under_test(test_member_path):
    global _test_stack
    _test_stack.append(test_member_path)
    print(f'test_stack: {_test_stack}')
    try:
        yield []
    finally:
        _test_stack.pop()


def from_scratch(path):
    """
    decorator that recursively finds all imports and mocks them all during function call.
    """

    def is_subpath(a, b):
        a_tokens = a.split('.')
        b_tokens = b.split('.')
        b = '.'.join(b_tokens[:len(a_tokens)])
        return len(a_tokens) <= len(b_tokens) and a == b


    # find all imports
    def decorator(member):
        member.__target_path__ = path

        @wraps(member)
        def wrapper(*args, **kwargs):
            current_test_member = _test_stack[-1]
            try:
                _, called_in_doctest = get_custom_doctests(current_test_member)
            except KeyError:
                called_in_doctest = ()

            with mock_patches(get_user_patches(member)):
                if is_subpath(current_test_member, member.__target_path__):
                    # this is the student member under test
                    # demote inputs, promote outputs
                    result = method_under_test_wrapper(member)(member)(*args, **kwargs)
                elif any([is_subpath(x, member.__target_path__) for x in called_in_doctest]):
                    # this is a student member used in the doctest
                    # demote inputs, promote outputs
                    result = method_under_test_wrapper(member)(member)(*args, **kwargs)
                else:
                    # this is a student member that is used by the student member under test
                    # demote inputs, demote outputs (should not be necessary because it isn't wrapped.)
                    result = member(*args, **kwargs)
            return result

        if inspect.isclass(inspect.unwrap(member)):
            # wrap each method of class
            for name, submember in inspect.getmembers(member):
                if inspect.isroutine(submember) and name in member.__dict__:
                    decorator = from_scratch(f'{path}.{name}')
                    setattr(member, name, decorator(submember))
            member.__wrapper__ = 'from_scratch'
            return member
        else:
            wrapper.__wrapper__ = 'from_scratch'
            return wrapper
    return decorator

def to_lol(a):
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(a, numpy.ma.MaskedArray):
        a = a.data  # ignore mask
    if isinstance(a, numpy.ndarray):
        return a.tolist()
    elif isinstance(a, tuple):
        return tuple(to_lol(list(a)))
    elif isinstance(a, list):
        return list(to_lol(ai) for ai in a)
    elif isinstance(a, dict):
        return {k: to_lol(v) for k, v in a.items()}
    return a

def is_lol(x):
    if isinstance(x, list):
        if len(x) > 0:
            if all([isinstance(xi, list) for xi in x]):
                return all([is_lol(xi) and len(xi) == len(x[0]) for xi in x])
            for xi in x:
                for t in [int, float, bool, str, complex]:
                    if numpy.issubdtype(type(xi), t):
                        break
                else:
                    return False
            return True
    return False

def to_array(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()

    if is_lol(x) or isinstance(x, LockedArray):
        return numpy.array(x)
    elif isinstance(x, tuple):
        return tuple([to_array(xi) for xi in x])
    elif isinstance(x, list):
        return [to_array(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()}
    return x

def to_locked(x):
    if isinstance(x, torch.Tensor):
        return LockedArray(x.detach().numpy())
    elif is_lol(x) or isinstance(x, numpy.ndarray):
        return LockedArray(x)
    elif isinstance(x, tuple):
        return tuple(to_locked(list(x)))
    elif isinstance(x, list):
        return [to_locked(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_locked(v) for k, v in x.items()}
    return x

def to_tensor(result, dtype=None):
    if isinstance(result, LockedArray):
        return torch.tensor(result, dtype=dtype)
    elif isinstance(result, numpy.ndarray):
        return torch.tensor(result, dtype=dtype)
    elif isinstance(result, torch.Tensor):
        return result.detach().clone().requires_grad_(result.requires_grad)
    elif isinstance(result, tuple):
        return tuple(to_tensor(ri, dtype=dtype) for ri in result)
    elif isinstance(result, list):
        return [to_tensor(ri, dtype=dtype) for ri in result]
    else:
        return result

def wrap_inputs(func, input_wrapper):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if input_wrapper is not None:
            args = input_wrapper(args)
            kwargs = input_wrapper(kwargs)
        return func(*args, **kwargs)

    wrapper.__wrapper__ = get_member_path(input_wrapper)
    return wrapper

def wrap_outputs(func, output_wrapper):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if output_wrapper is not None:
            result = output_wrapper(result)
        return result

    wrapper.__wrapper__ = get_member_path(output_wrapper)
    return wrapper

def wrap_inputs_and_outputs(func, input_wrapper, output_wrapper, use_out=False):
    return wrap_outputs(wrap_inputs(func, input_wrapper), output_wrapper)

def numpy_transform_inputs_and_outputs(func):
    """
    This transforms the inputs and outputs of our list-of-lists function when testing so that it works with numpy/torch
    functions.
    """
    def copy_into(a, b):
        if isinstance(b, numpy.ndarray):
            a = numpy.array(a)
            b.ravel()[:] = a.ravel()[:]
            # assume a and b have the same shape
            if b.shape != a.shape:
                # reshaping creates a view and no long "is" the same object.
                b = b.reshape(a.shape)
        elif isinstance(b, list):
            # this is a spurious hit for the output being the same object as an input.

            # this is actually a mistake. numpy doesn't have an "out" that can be a list.
            # just make it an array
            b.clear()
            b.extend(a)
            b = numpy.array(b)
        return b

    def transform_output_and_honor_out(a, result_type, args, kwargs, original_args, original_kwargs):
        # args, kwargs = transformed_arguments
        # original_args, original_kwargs = original_arguments
        # transform output honoring the 'out' argument
        if isinstance(a, tuple):
            return tuple([transform_output_and_honor_out(ai, result_type, args, kwargs, original_args, original_kwargs) for ai in a])
        elif isinstance(a, (list, numpy.ndarray)):
            # If the result is a reference to one of the inputs, don't make a copy, fill in the original
            # argument with the result values
            for i, arg in enumerate(args):
                if a is arg:
                    a = copy_into(arg, original_args[i])
                    break
            else:
                for k, v in kwargs.items():
                    if a is v:
                        a = copy_into(v, original_kwargs[k])
                        break
                else:
                    a = numpy.array(a, dtype=result_type)
        elif is_numeric(a):
            if result_type is not None:
                a = result_type(a)
        return a

    def find_out(out, transformed_args, original_args):
        if isinstance(out, tuple):
            # look for each output
            return tuple([find_out(out_i, transformed_args, original_args) for out_i in out])
        elif isinstance(out, (list, numpy.ndarray)):
            if out is transformed_args:
                return original_args
            elif isinstance(transformed_args, (tuple, list)):
                for ai, bi in zip(transformed_args, original_args):
                    a_out = find_out(out, ai, bi)
                    if a_out is not None:
                        return a_out
            elif isinstance(transformed_args, dict):
                for k in transformed_args.keys():
                    ai = transformed_args[k]
                    bi = original_args[k]
                    a_out = find_out(out, ai, bi)
                    if a_out is not None:
                        return a_out
        return None


    def update_result_type(bound_args):
        result_type = bound_args.arguments.get('dtype', None)
        args = bound_args.args
        if result_type is None and len(args) > 0 and isinstance(args[0], numpy.ndarray) and numpy.issubdtype(args[0].dtype, numpy.unsignedinteger):
            result_type = numpy.uint
        if 'dtype' in bound_args.arguments:
            bound_args.arguments['dtype'] = result_type
        return bound_args

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_args = args
        original_kwargs = kwargs

        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()

        bound_args = update_result_type(bound_args)

        dtype = bound_args.arguments.get('dtype', None)

        args = to_lol(bound_args.args)
        kwargs = to_lol(bound_args.kwargs)

        result = func(*args, **kwargs)
        result = transform_output_and_honor_out(result, dtype, args, kwargs, original_args, original_kwargs)
        return result

    wrapper.__wrapper__ = 'numpy_transform_inputs_and_outputs'
    return wrapper

class LockedArray(numpy.ndarray):
    ALLOWED_METHODS = ['copy', 'ravel', '__getattr__', '__getattribute__', '__getitem__', '__str__', '__repr__', 'view']
    ALLOWED_ATTRS = []
    OPERATORS = [x for x in dir(operator) if x.startswith('__') and x.endswith('__')]
    ALLOWED_OPERATORS = ['__sub__', '__getitem__']
    def __new__(cls, input_array):
        return numpy.asarray(input_array).view(cls)
    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ufunc_path = get_member_path(ufunc)
        if ufunc_path not in allowed_ufuncs:
            raise TypeError(f'Operation "{ufunc_path}" is not allowed on LockedArray.')
        inputs = tuple(numpy.array(x) if isinstance(x, LockedArray) else x for x in inputs)
        if 'out' in kwargs and kwargs['out'] is not None:
            kwargs['out'] = tuple(numpy.array(x) if isinstance(x, LockedArray) else x for x in kwargs['out'])

        results = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if results is not NotImplemented:
            if method == 'at':
                return None
            if isinstance(results, tuple):
                return tuple(LockedArray(x) for x in results)
            return LockedArray(results)

        return NotImplemented

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        # specifically disallow operators
        if name in ['ALLOWED_METHODS', 'ALLOWED_ATTRS', 'OPERATORS', 'ALLOWED_OPERATORS']:
            return attr
        elif callable(attr) and name not in self.ALLOWED_METHODS + allowed_ufuncs + allowed_methods:
            raise AttributeError(
                f'Method .{name} is disabled for LockedArray.'
            )
        elif not callable(attr) and name not in self.ALLOWED_ATTRS + allowed_attrs:
            raise AttributeError(
                f'Attribute .{name} is disabled for LockedArray.'
            )
        return attr
    def __str__(self):
        return numpy.array(self).__str__()
    def __repr__(self):
        return numpy.array(self).__repr__()

def torch_transform_inputs_and_outputs(func):
    """
    This transforms the inputs and outputs of our list-of-lists function when testing so that it works with numpy/torch
    functions.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        result_type = bound_args.arguments.get('dtype', None)

        args = to_locked(bound_args.args)
        kwargs = to_locked(bound_args.kwargs)
        result = func(*args,  **kwargs)
        result = to_tensor(result, result_type)
        return result

    wrapper.__wrapper__ = 'torch_transform_inputs_and_outputs'
    return wrapper

def import_from_path(module_name, file_path):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    if module_name in sys.modules:
        return sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

def get_module(module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    else:
        return importlib.import_module(module_name)

def get_member(member_path, return_class=False):
    k = -1
    while True:
        try:
            # getting a function
            module_name, member_name = split_member_path(member_path, k)
            if len(module_name) == 0:
                raise ValueError(f'Failed to get member {member_path}')
            module = get_module(module_name)
            member = module
            class_ = None
            for token in member_name.split('.'):
                member = unwrap(member)
                if inspect.isclass(member):
                    class_ = member
                    if return_class:
                        break
                    _, member = inspect.getmembers(member, lambda x: inspect.isroutine(x) and x.__name__ == token)[0]
                else:
                    member = getattr(member, token)
            if return_class:
                return class_
            else:
                return member
        except ModuleNotFoundError:
            k -= 1

TREE = {}
def get_target(member_path):
    if member_path in TREE:
        return TREE[member_path]
    elif member_path in TREE.values():
        return member_path
    else:
        raise KeyError(f'{member_path} not present.')

def set_target(member_path, target):
    """
    If the tree doesn't have the member_path or it points to itself, replace the value with target.
    When it points to itself, its should be a student implemented helper function
    """
    if member_path in TREE:
        if TREE[member_path] == member_path:
            TREE[member_path] = target
    else:
        TREE[member_path] = target

def split_member_path(path, k=-1):
    tokens = path.split('.')
    module_name = '.'.join(tokens[:k])
    member_name = '.'.join(tokens[k:])
    return module_name, member_name

def get_member_imports(member):
    source = textwrap.dedent(inspect.getsource(member))
    return get_imports(source)

def get_source_calls(member):
    member_path = get_member_path(member)
    source = inspect.getsource(member)
    closure_vars = inspect.getclosurevars(inspect.unwrap(member))
    search_scope = {**closure_vars.globals, **closure_vars.builtins}

    # 2. Parse the source into an AST
    tree = ast.parse(textwrap.dedent(source))

    # 3. Walk through the nodes and find imports
    # how to walk: https://mvdwoord.github.io/exploration/2017/08/18/ast_explore.html
    # types of nodes to filter: https://docs.python.org/3/library/ast.html

    dependencies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in search_scope:
                    obj = search_scope[node.func.id]
                    dependency_module_name = get_module_name(inspect.getmodule(obj))
                    dependency_path = f'{dependency_module_name}.{obj.__name__}'
                    # ignore builtins and bootstrap functions. Do not recurse (could be handled by caching test results.)
                    if dependency_module_name not in ('builtins', 'from_scratch.from_scratch') and dependency_path != member_path:
                        dependencies.append(dependency_path)
                        dependencies.extend(get_member_dependencies(dependency_path))
        elif isinstance(node, ast.BinOp):
            pass
    return dependencies


def get_member_dependencies(member_path):
    # 1. Get the source code of the function
    # https://docs.python.org/3/library/inspect.html#inspect.getsourcefile
    # need to unwrap function to find its function calls

    member = get_member(member_path)
    member = unwrap(member)

    dependency_module_name = get_module_name(member)


    dependencies = []

    if is_allowed(dependency_module_name):
        return dependencies

    dependencies += get_member_imports(member)

    if inspect.isclass(member):
        # get inherited class dependencies
        for class_ in inspect.getmro(member)[1:]:
            path = get_member_path(class_)
            if is_allowed(path):
                break
            dependencies.append(path)

        # look at members
        for class_member_name, class_member in inspect.getmembers(member):
            # get locally defined (not inherited) methods and classes
            if class_member_name in member.__dict__:
                if inspect.isroutine(class_member) or inspect.isroutine(class_member):
                    path = get_member_path(class_member)
                    # dependencies.append(path)
                    # dependencies += get_member_dependencies(path)
                    dependencies += get_source_calls(class_member)

    elif inspect.isroutine(member):
        dependencies += get_source_calls(member)

    return list(set(dependencies))

def get_imports(source):
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            imports.extend([f"{node.module}.{name.name}" for name in node.names])
    return imports

def get_file_imports(file_name):
    with open(file_name, 'r') as f:
        source = f.read()
    return get_imports(source)

def unwrap(f):
    while hasattr(f, '__wrapped__'):
        f = f.__wrapped__
    return f

payloads = None
class_payloads = None
allowed_ufuncs = []
allowed_attrs = []
allowed_methods = []


def pytest_generate_tests(metafunc):
    """
    Gather all functions defined in all python files excluding from_scratch.py
    """
    print('Gathering tests')
    if "test_member_path" in metafunc.fixturenames:
        # find all imports in user files
        all_imports = []
        for file_name in os.listdir('.'):
            if file_name.lower().endswith('.py'):
                file_imports = get_file_imports(file_name)
                file_imports = list(set(file_imports))
                file_imports = [x for x in file_imports if not is_allowed(x)]
                for file_import in file_imports:
                    print(f'{file_name} imports {file_import}')
                all_imports.extend(file_imports)
        all_imports = list(set(all_imports))

        # find all python files in current
        names = []
        for file_name in os.listdir('.'):
            if file_name.lower().endswith('.py') and file_name != os.path.split(__file__)[-1]:
                print(f'Gathering tests for {file_name}')
                # construct import tree
                module_name = file_name.split('.')[0]
                mod = import_from_path(module_name, file_name)
                members_list = inspect.getmembers(mod, lambda x: inspect.isroutine(x) or inspect.isclass(x))

                for member_name, member in members_list:
                    unwrapped_member = unwrap(member)
                    member_module = inspect.getmodule(unwrapped_member)

                    if member_module.__file__ != __file__:
                        member_path = get_member_path(member)

                        if hasattr(member, '__target_path__'):
                            # set the target path to point to this member too.
                            set_target(member.__target_path__, member_path)
                            # adding test for it
                            names.append(member.__target_path__)
                            print(f'{file_name}: {member_path} implements {member.__target_path__}')
                        else:
                            set_target(member_path, member_path)

        for ufunc in UFUNC_LIST:
            try:
                print(f'Checking prerequisite: numpy.{ufunc}')
                test_member(ufunc)
                allowed_ufuncs.append(ufunc)
            except:
                print(f'Failed numpy.{ufunc}')
        for property in ['size', 'shape', 'ndim', 'dtype', 'real', 'image']:
            try:
                print(f'Checking prerequisite: numpy.{property}')
                test_member(f'numpy.{property}')
                allowed_attrs.append(property)
            except:
                print(f'Failed numpy.{property}')
        try:
            print(f'Checking prerequisite: numpy.transpose')
            test_member('numpy.transpose')
            allowed_attrs.append('T')
        except:
            print(f'Failed numpy.transpose')

        for method in ['reshape', 'ravel']:
            try:
                print(f'Checking prerequisite: numpy.{method}')
                test_member(f'numpy.{method}')
                allowed_methods.append(method)
            except:
                print(f'Failed numpy.{method}')

        for method in [x for x in dir(numpy.array([1, 2, 3])) if not x.startswith('_')]:
            try:
                print(f'Checking prerequisite: numpy.{method}')
                test_member(f'numpy.{method}')
                allowed_methods.append(method)
            except:
                print(f'Failed numpy.{method}')

        names = TESTS or list(set(all_imports + names))
        metafunc.parametrize("test_member_path", names, ids=names)



@contextmanager
def mock_patches(patches=()):
    for p in patches:
        p.start()

    try:
        yield patches

    finally:
        for p in patches:
            p.stop()


def run_example(source, globs, patches=()):
    with mock_patches(patches):
        try:
            return eval(source, globs)
        except SyntaxError:
            return exec(source, globs)
        except Exception as e:
            # print(traceback.format_exc())
            return e  # str(e)

def check_result(source, correct, student):
    s = f'\n>>> {source}'
    s += f'Expected:\n{repr(correct)}\n'
    s += f'Got:\n{repr(student)}\n'
    try:
        m = matches(student, correct)
    except Exception as e:
        raise AssertionError(f'matches raises exception {e}:\n{s}')
    if m < 0:
        s += '*** Failed ***\n'
        raise AssertionError(s)
    elif m == 0:
        s += '*** Trivial ***\n'
    else:
        s += '*** Success ***\n'
    return s

def seed(a=None):
    """
    Helper seed function to reset all seeds.
    """
    if a is None:
        a = int(datetime.now().timestamp())
    torch.random.manual_seed(a)
    numpy.random.seed(a)
    random.seed(a)
    return a

def deep_copy(d):
    # deep copy of globals dict
    result = {}
    for k, v in d.items():
        module = inspect.getmodule(v)
        if module is not None:
            if module.__name__ == 'torch.nn.modules.container':
                v = copy.deepcopy(v)
        result[k] = v
    return result

def _my_shuffle(x):
    # to work with arange must not use arange...
    if isinstance(x, list):
        return _my_shuffle(numpy.array(x)).tolist()
    elif isinstance(x, numpy.ndarray):
        if x.size > 1:
            idx = numpy.random.permutation(range(x.size))
            while numpy.all(idx[1:] > idx[:-1]):
                idx = numpy.random.permutation(x.size)
            x[:] = x.ravel()[idx]
            return x
    elif isinstance(x, torch.Tensor):
        if x.nelement() > 1:
            rng = torch.tensor(range(x.nelement()))
            idx = torch.randperm(x.nelement())
            while torch.all(torch.eq(idx, rng)):
                idx = torch.randperm(x.nelement())
            return x.view(-1)[idx].view(x.size())
    return x


def run_examples(examples, name=None, shuffle=False):
    globs = {'torch': torch, 'np': numpy, 'random': random, 'nn': torch.nn, 'math': math, 'gpt': gpt, '_my_shuffle': _my_shuffle}
    student_globs = deep_copy(globs)
    s = '\n\n'

    sources = [example.source for example in examples]
    student_patches = get_library_to_student_patches(name)

    nontrivial = 0
    for source in sources:
        if source == 'a = np.zeros((2, 512*512), dtype=np.float32)\n':
            # fix slow numpy.var test
            source = 'a = np.zeros((2, 16*16), dtype=np.float32)\n'
        old_globs = deep_copy(student_globs)
        seed_ = seed()
        student = run_example(source, student_globs, patches=student_patches)  # ignore globals updated by student code.
        seed(seed_)
        correct = run_example(source, globs)  # update globals for next example
        cr = check_result(source, correct, student)
        nontrivial += int('*** Success ***' in cr)
        print(cr)
        s += cr
        # if it's an assignment statement something global changes, so check it:
        modified_globs = {k: v for k, v in student_globs.items() if
                          (k not in old_globs or not numpy.array_equal(v, old_globs[k])) and not k.startswith('__')}
        for k, v in modified_globs.items():
            modified_source = f'{k}\n'
            seed_ = seed()
            student = run_example(modified_source, student_globs,
                                  patches=student_patches)  # ignore globals updated by student code.
            seed(seed_)
            correct = run_example(modified_source, globs)  # update globals for next example
            cr = check_result(modified_source, correct, student)
            print(cr)
            s += cr

        if shuffle:
            # check if there are new array variables and shuffle them
            modified_globs = {k: v for k, v in student_globs.items() if (k not in old_globs or not numpy.array_equal(v, old_globs[k])) and not k.startswith('__')}
            for k, v in modified_globs.items():
                shuffle_source = None
                # handle list of tuples of arrays.
                if isinstance(v, (list, numpy.ndarray, torch.Tensor)) and type(v) is type(globs[k]):
                    shuffle_source = f'_my_shuffle({k})\n'

                if shuffle_source is not None:
                    seed_ = seed()
                    student = run_example(shuffle_source, student_globs, patches=student_patches)  # ignore globals updated by student code.
                    seed(seed_)
                    correct = run_example(shuffle_source, globs)  # update globals for next example
                    cr = check_result(shuffle_source, correct, student)
                    print(cr)
                    s += cr
                    shuffle_source = f'{k}\n'
                    seed_ = seed()
                    student = run_example(shuffle_source, student_globs, patches=student_patches)  # ignore globals updated by student code.
                    seed(seed_)
                    correct = run_example(shuffle_source, globs)  # update globals for next example
                    # following line fails because 'y' gets shuffled differently?
                    cr = check_result(shuffle_source, correct, student)
                    print(cr)
                    s += cr
    if nontrivial == 0:
        raise AssertionError(s + '*** Trivial Success ***\n')
    return s

def matches(actual, expected):
    """
    a = student, b = correct
    Returns 1=matches, 0=trivial match, -1 does not match
    """
    def compare(ai, ei):
        for t in [AxesImage, Line2D, BarContainer, Rectangle, DataLoader]:
            if isinstance(ai, t) or isinstance(ei, t):
                return 0

        # the following fixes the situation where we are storing the value of a loss function and returning the loss
        # function when it gets called. This makes it easier to backprop from the loss function rather than creating
        # an entirely new tensor class to hold values that can backprop. The problem is that sometimes we want the
        # loss function object for testing and sometimes its value.
        # The following condition checks if ai is a loss function but bi is not and gets ai's value for comparison in
        # that case.
        if 'loss' in getattr(ai, '__target_path__', '').lower() and 'loss' not in get_shorter_member_path(ei.__class__).lower():
            # handles a loss function with a "value"
            ai = ai.item()


        # if we're expecting an exception make sure they're the same type
        if isinstance(ei, Exception):
            return int(type(ai) == type(ei)) * 2 - 1

        # if we weren't expecting an exception, but actually got one
        if isinstance(ai, Exception):
            return -1


        if isinstance(ai, dict) or isinstance(ei, dict):
            if '__builtins__' in ai or '__builtins__' in ei:
                return 0
            if ai.keys() != ei.keys():
                return -1
            return min([compare(aij, bij) for aij, bij in zip(ai.values(), ei.values())])

        if isinstance(ai, list) or isinstance(ei, list):
            if len(ai) != len(ei):
                return -1
            return min([compare(aij, bij) for aij, bij in zip(ai, ei)])

        if ai is None and ei is None:
            # both are None
            return 0
        elif ai is None or ei is None:
            # only one is None
            return -1

        if isinstance(ai, torch.nn.Module) or isinstance(ei, torch.nn.Module):
            # if it matches, a trivial match
            return int(repr(ai) == repr(ei)) - 1

        if isinstance(ai, torch.optim.Optimizer) or isinstance(ei, torch.optim.Optimizer):
            # if it matches, a trivial match
            return int(repr(ai) == repr(ei)) - 1

        if isinstance(ai, torch.nn.Parameter) or isinstance(ei, torch.nn.Parameter):
            ai = ai.data
            ei = ei.data

        if isinstance(ai, LockedArray):
            ai = numpy.array(ai)
        if isinstance(ei, LockedArray):
            ei = numpy.array(ei)

        if isinstance(ai, torch.Tensor):
            ai = ai.detach().numpy()
        if isinstance(ei, torch.Tensor):
            ei = ei.detach().numpy()

        if not numpy.all(ai == ei) and not numpy.allclose(ai, ei, equal_nan=True, atol=1e-6):
            return -1
        return 1

    actual = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)
    results = []
    if len(actual) != len(expected):
        return -1

    for ai, bi in zip(actual, expected):
        results.append(compare(ai, bi))
    if any([r < 0 for r in results]):
        return -1
    elif any([r > 0 for r in results]):
        return 1
    else:
        return 0


def get_doctests(name):
    function = get_member(name)
    doc_string = function.__doc__
    return doc_string or ''


def run_doctests(name, doc_test, shuffle=False):
    # run doc tests
    function = get_member(name)
    module = inspect.getmodule(function)

    # mock the appropriate random library
    base_module_name = module.__name__.split('.')[0]
    with mock_patches(get_doctest_patches(base_module_name)):
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore', category=RuntimeWarning)
            return run_examples(doc_test.examples, name, shuffle=shuffle)


def memoize(func):
    results = {}
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        key = '_'.join(bound_args.arguments.values())
        if key not in results:
            try:
                results[key] = func(*args, **kwargs)
            except Exception as e:
                results[key] = e
        result = results[key]
        print(result)
        if isinstance(result, Exception):
            raise result
        return result

    wrapper.__wrapper__ = 'memoize'
    return wrapper

def get_member_path(member):
    return get_shorter_member_path(member)

def get_shorter_member_path(member):
    base_module_name = get_module_name(member)
    member_name = getattr(member, '__qualname__', getattr(member, '__name__', None))
    shortest_path = f'{base_module_name}.{member_name}'
    tokens = base_module_name.split('.')
    for i in range(1, len(tokens)+1):
        parent_module_name = '.'.join(tokens[:i])
        try:
            module = get_module(parent_module_name)
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                except TypeError:
                    continue
                except Exception as e:
                    continue
                if attr is member:
                    shortest_path = f'{parent_module_name}.{attr_name}'
                    break
            else:
                continue
            break
        except (ImportError, AttributeError):
            continue
    return shortest_path

def get_module_name(member):
    module = inspect.getmodule(member)
    return module.__name__ if module else 'builtins'

def within_class(member_path):
    return get_member(member_path, return_class=True) is not None

def is_student_member(member_path):
    return member_path.split('.')[0] + '.py' in [x.lower() for x in os.listdir()]

def is_class(member_path):
    return inspect.isclass(inspect.unwrap(get_member(member_path)))

@memoize
def check_member(test_member_path):
    print(f'Checking member {test_member_path}')
    if not is_allowed(test_member_path) and not is_dtype(test_member_path):
        # check that it has a target implementation.
        target_path = get_target(test_member_path)
        if is_student_member(target_path):
            # if it's not a student member, don't test it.
            test_member_dependencies = get_member_dependencies(target_path)
            print(test_member_path, 'depends on:', test_member_dependencies)

            for dependency in test_member_dependencies:
                test_member(dependency)

            if within_class(test_member_path):
                return f'{test_member_path} is a method without doctests.\n'
            elif is_student_member(test_member_path):
                return f'{test_member_path} is a student member without doctests.\n'
            elif is_class(test_member_path):
                doc_string, _ = get_custom_doctests(test_member_path)
                doc_test = doctest.DocTestParser().get_doctest(doc_string, {}, 'example_test', None, 0)
                return run_doctests(test_member_path, doc_test)
            else:
                try:
                    # if zero doctests or all doctests are trivial raises assertion error
                    doc_string = get_doctests(test_member_path)
                    doc_test = doctest.DocTestParser().get_doctest(doc_string, {}, 'example_test', None, 0)
                    s = run_doctests(test_member_path, doc_test)
                    s += run_doctests(test_member_path, doc_test, shuffle=True)
                    return s
                except AssertionError as e:
                    if '*** Trivial Success ***' in str(e):
                        doc_string, _ = get_custom_doctests(test_member_path)
                        doc_test = doctest.DocTestParser().get_doctest(doc_string, {}, 'example_test', None, 0)
                        return run_doctests(test_member_path, doc_test)
                    else:
                        raise e
        else:
            raise AssertionError(f'{target_path} is not a student member and is not allowed.')
    else:
        return f'{test_member_path} is allowed.\n'

def test_member(test_member_path):
    # TODO: make sure no imports outside function calls.
    # TODO: mock all ndarray methods except those associated with passed tests
    print(f'\n<Testing {test_member_path}>')
    with under_test(test_member_path):
        check_member(test_member_path)
    print(f'\n</Testing {test_member_path}>')

def get_custom_doctests(name):
    module_name = name.split('.')[0]
    check_path = 'from_scratch.' + module_name + '_checks' + '.check_' + name.lower().replace('.', '_')
    try:
        return get_member(check_path)()
    except AttributeError as e:
        raise KeyError(f'Tests for {name} not found')


    return get_member(check_path)()

def main():
    pytest.main([__file__, '-ra', '-s'])

if __name__ == '__main__':
    main()

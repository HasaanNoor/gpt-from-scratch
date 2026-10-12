# These are RMP's course materials for CS 5750: Demystifying LLMs.
# Public sharing of course materials is prohibited.

import textwrap

def check_gpt_logisticregression():
    docstring = textwrap.dedent("""\
    >>> m = gpt.LogisticRegression(5)
    >>> m
    >>> x = torch.randn(3, 5, requires_grad=True)
    >>> output = m(x)
    >>> d_out = torch.randn(3, 1)
    >>> total = (output * d_out).sum()
    >>> if hasattr(m, '__target_path__'):
    ...     d_in = m.backward(d_out)
    ... else:
    ...     total.backward()
    ...     d_in = x.grad
    >>> d_in
    >>> list(m.parameters())
    >>> [x.grad for x in m.parameters()]
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_gpt_layernorm():
    docstring = textwrap.dedent("""\
    >>> m = gpt.LayerNorm(5)
    >>> m
    >>> list(m.parameters())
    >>> x = torch.randn(3, 5, requires_grad=True)
    >>> x
    >>> output = m(x)
    >>> d_out = torch.randn(3, 5)
    >>> total = (output * d_out).sum()
    >>> if hasattr(m, '__target_path__'):
    ...     d_in = m.backward(d_out)
    ... else:
    ...     total.backward()
    ...     d_in = x.grad
    >>> list(m.parameters())
    >>> [x.grad for x in m.parameters()]
    >>> d_in
    """)
    dependent_members = ()
    return docstring, dependent_members

def check_gpt_multiheadattention():
    docstring = textwrap.dedent("""\
    >>> emb_dim = 30
    >>> context_length = 32
    >>> n_heads = 5
    >>> drop_rate = 0.1
    >>> qkv_bias = False
    >>> m = gpt.MultiHeadAttention(emb_dim, emb_dim, context_length, drop_rate, n_heads, qkv_bias=qkv_bias)
    >>> m
    >>> list(m.parameters())
    >>> b = 1
    >>> num_tokens = 4
    >>> input = torch.randn(b, num_tokens, emb_dim, requires_grad=True)
    >>> output = m(input)
    >>> output
    >>> d_out = torch.randn(output.size())
    >>> # check intermediate results
    >>> if hasattr(m, '__target_path__'):
    ...     m.backward(d_out)
    ... else:
    ...     (d_out * output).sum().backward()
    >>> [x.grad for x in m.out_proj.parameters()]
    >>> [x.grad for x in m.W_value.parameters()]
    >>> if hasattr(m, '__target_path__'):
    ...     d_context_vec = m.d_context_vec
    ... else:
    ...     d_context_vec = m.context_vec.grad
    >>> d_context_vec
    >>> if hasattr(m, '__target_path__'):
    ...     d_attn_weights = m.d_attn_weights
    ... else:
    ...     d_attn_weights = m.attn_weights.grad
    >>> d_attn_weights
    >>> if hasattr(m, '__target_path__'):
    ...     d_softmax_out = m.d_softmax_out
    ... else:
    ...     d_softmax_out = m.softmax_out.grad
    >>> d_softmax_out
    >>> if hasattr(m, '__target_path__'):
    ...     d_scaled_attn = m.d_scaled_attn
    ... else:
    ...     d_scaled_attn = m.scaled_attn.grad
    >>> d_scaled_attn
    >>> if hasattr(m, '__target_path__'):
    ...     d_attn_scores = m.d_attn_scores
    ... else:
    ...     d_attn_scores = m.attn_scores.grad
    >>> d_attn_scores
    >>> [x.grad for x in m.W_query.parameters()]
    >>> [x.grad for x in m.W_key.parameters()]
    >>> d_in
    """)
    dependent_members = ()  # ('torch.nn.Linear', 'torch.nn.Dropout', 'torch.nn.Softmax')
    return docstring, dependent_members

def check_gpt_gelu():
    docstring = textwrap.dedent("""\
    >>> m = gpt.GELU()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> output = m(input)
    >>> output
    >>> d_out = torch.randn(output.size())
    >>> if hasattr(m, '__target_path__'):
    ...     d_in = m.backward(d_out)
    ... else:
    ...     (d_out * output).sum().backward()
    ...     d_in = input.grad
    >>> d_in
    """)
    dependent_members = ()
    return docstring, dependent_members

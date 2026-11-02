import torch
import torch.nn as nn


# A logistic regression example:
class LogisticRegression(nn.Module):
    """
    >>> module = raschka.gpt.LogisticRegression(5)
    >>> input = torch.randn(3, 5, requires_grad=True)
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
    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))


def logistic_regression_check():
    num_inputs = 3
    num_samples = 5
    x = torch.randn(num_samples, num_inputs)
    y = (torch.randn(num_samples, 1) > 0).float()
    model = LogisticRegression(num_inputs)
    criterion = nn.BCELoss()
    yhat = model(x)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(10000):
        model.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        print(loss.item())
        loss.backward()
        optimizer.step()

# logistic_regression_check()

# A logistic regression example:
class SoftmaxRegression(nn.Module):
    """
    >>> module = raschka.gpt.SoftmaxRegression(5, 4)
    >>> input = torch.randn(3, 5, requires_grad=True)
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
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.softmax = nn.Softmax()
    def forward(self, x):
        return self.softmax(self.linear(x))


###########################################################################################
# The following is from Sebastian Raschka's GPT from scratch:
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py
###########################################################################################

# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

import tiktoken
from torch.utils.data import Dataset, DataLoader


#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """
    >>> tokenizer = tiktoken.get_encoding("gpt2")
    >>> text = '''I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)
    ...
    ... "The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. "Of course it's going to send the value of my picture 'way up; but I don't think of that, Mr. Rickham--the loss to Arrt is all I think of." The word, on Mrs. Thwing's lips, multiplied its _rs_ as though they were reflected in an endless vista of mirrors. And it was not only the Mrs. Thwings who mourned. Had not the exquisite Hermia Croft, at the last Grafton Gallery show, stopped me before Gisburn's "Moon-dancers" to say, with tears in her eyes: "We shall not look upon its like again"?
    ...
    ... '''
    >>> dataset = raschka.gpt.GPTDatasetV1(text, tokenizer, 3, 3)
    >>> len(dataset)
    >>> list(dataset)
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    >>> text = '''I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)
    ...
    ... "The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. "Of course it's going to send the value of my picture 'way up; but I don't think of that, Mr. Rickham--the loss to Arrt is all I think of." The word, on Mrs. Thwing's lips, multiplied its _rs_ as though they were reflected in an endless vista of mirrors. And it was not only the Mrs. Thwings who mourned. Had not the exquisite Hermia Croft, at the last Grafton Gallery show, stopped me before Gisburn's "Moon-dancers" to say, with tears in her eyes: "We shall not look upon its like again"?
    ...
    ... '''
    >>> dataloader = raschka.gpt.create_dataloader_v1(text, batch_size=2, max_length=5, stride=5)
    >>> len(dataloader)
    >>> list(dataloader)
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, generator=torch.default_generator)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """
    >>> emb_dim, context_length, drop_rate, num_heads, qkv_bias = 8, 5, 0.5, 4, True
    >>> num_batches, num_tokens = 2, 3
    >>> module = raschka.gpt.MultiHeadAttention(emb_dim, emb_dim, context_length, drop_rate, num_heads, qkv_bias)
    >>> input = torch.randn(num_batches, num_tokens, emb_dim, requires_grad=True)
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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        scaled_attn = attn_scores / keys.shape[-1]**0.5

        softmax_out = torch.softmax(scaled_attn, dim=-1)

        attn_weights = self.dropout(softmax_out)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    >>> module = raschka.gpt.LayerNorm(5)
    >>> input = torch.randn(2, 3, 5, requires_grad=True)
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
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    >>> module = raschka.gpt.GELU()
    >>> input = torch.randn(3, 5, requires_grad=True)
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    >>> cfg = dict(emb_dim=6, context_length=4, n_heads=3, drop_rate=0.5, qkv_bias=False)
    >>> module = raschka.gpt.FeedForward(cfg)
    >>> input = torch.randn(3, cfg['emb_dim'], requires_grad=True)
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
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    >>> cfg = dict(emb_dim=6, context_length=4, n_heads=3, drop_rate=0.5, qkv_bias=False)
    >>> module = raschka.gpt.TransformerBlock(cfg)
    >>> input = torch.randn(2, cfg['context_length'], cfg['emb_dim'], requires_grad=True)
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
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut1 = nn.Dropout(cfg["drop_rate"])
        self.drop_shortcut2 = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut1(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut2(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """
    >>> cfg = dict(emb_dim=6, context_length=4, n_heads=3, drop_rate=0.5, qkv_bias=False, vocab_size=6, n_layers=2)
    >>> module = raschka.gpt.GPTModel(cfg).train()
    >>> input = torch.randint(0, cfg['vocab_size'], (2, cfg['context_length']))
    >>> list(module.parameters())
    >>> output = module(input)
    >>> [p.grad for p in module.parameters()]
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    >>> cfg = dict(emb_dim=6, context_length=4, n_heads=3, drop_rate=0.5, qkv_bias=False, vocab_size=6, n_layers=2)
    >>> model = raschka.gpt.GPTModel(cfg).eval()
    >>> encoded_tensor = torch.randint(0, cfg['vocab_size'], (2, cfg["context_length"]))
    >>> out = raschka.gpt.generate_text_simple(
    ...     model=model,
    ...     idx=encoded_tensor,
    ...     max_new_tokens=10,
    ...     context_size=cfg["context_length"]
    ... )
    """
    idx = torch.tensor(idx)
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    """
    >>> raschka.gpt.main()
    """
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.5,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    # torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
    return decoded_text

if __name__ == "__main__":
    main()
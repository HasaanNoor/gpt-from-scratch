import textwrap

def check_raschka_gpt_train_train_model_simple():
    docstring = textwrap.dedent("""\
    >>> GPT_CONFIG = dict(vocab_size=50257, context_length=5, emb_dim=6, n_heads=3, n_layers=2, drop_rate=0.1, qkv_bias=False)
    >>> OTHER_SETTINGS = dict(learning_rate=5e-4, num_epochs=10, batch_size=2, weight_decay=0.1)
    >>> device = torch.device('cpu')
    >>> model = raschka.gpt.GPTModel(GPT_CONFIG)
    >>> tokenizer = tiktoken.get_encoding("gpt2")
    >>> text_data = '''I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)
    ...
    ... "The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. "Of course it's going to send the value of my picture 'way up; but I don't think of that, Mr. Rickham--the loss to Arrt is all I think of." The word, on Mrs. Thwing's lips, multiplied its _rs_ as though they were reflected in an endless vista of mirrors. And it was not only the Mrs. Thwings who mourned. Had not the exquisite Hermia Croft, at the last Grafton Gallery show, stopped me before Gisburn's "Moon-dancers" to say, with tears in her eyes: "We shall not look upon its like again"?
    ...
    ... '''
    >>> train_data = text_data[:len(text_data)//2]
    >>> val_data = text_data[len(text_data)//2:]
    >>> kwargs = dict(batch_size=OTHER_SETTINGS['batch_size'], max_length=GPT_CONFIG['context_length'], stride=GPT_CONFIG['context_length'], drop_last=True, shuffle=True, num_workers=0)
    >>> train_loader = raschka.gpt.create_dataloader_v1(train_data, **kwargs)
    >>> val_loader = raschka.gpt.create_dataloader_v1(val_data, **kwargs)
    >>> optimizer = torch.optim.AdamW(
    ...     model.parameters(), lr=OTHER_SETTINGS["learning_rate"], weight_decay=OTHER_SETTINGS["weight_decay"]
    ... )
    >>> train_losses, val_losses, tokens_seen = raschka.gpt_train.train_model_simple(
    ...     model, train_loader, val_loader, optimizer, device,
    ...     num_epochs=OTHER_SETTINGS["num_epochs"], eval_freq=5, eval_iter=1,
    ...     start_context="Every effort moves you", tokenizer=tokenizer
    ... )
    """)
    dependent_members = ('raschka.gpt.GPTModel', 'torch.optim.AdamW')
    return docstring, dependent_members

def check_raschka_gpt_train_calc_loss_loader():
    docstring = textwrap.dedent("""\
    >>> GPT_CONFIG = dict(vocab_size=50257, context_length=5, emb_dim=6, n_heads=3, n_layers=2, drop_rate=0.1, qkv_bias=False)
    >>> OTHER_SETTINGS = dict(learning_rate=5e-4, num_epochs=10, batch_size=2, weight_decay=0.1)
    >>> device = torch.device('cpu')
    >>> text_data = '''I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)
    ...
    ... "The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. "Of course it's going to send the value of my picture 'way up; but I don't think of that, Mr. Rickham--the loss to Arrt is all I think of." The word, on Mrs. Thwing's lips, multiplied its _rs_ as though they were reflected in an endless vista of mirrors. And it was not only the Mrs. Thwings who mourned. Had not the exquisite Hermia Croft, at the last Grafton Gallery show, stopped me before Gisburn's "Moon-dancers" to say, with tears in her eyes: "We shall not look upon its like again"?
    ...
    ... '''
    >>> kwargs = dict(batch_size=OTHER_SETTINGS['batch_size'], max_length=GPT_CONFIG['context_length'], stride=GPT_CONFIG['context_length'], drop_last=True, shuffle=True, num_workers=0)
    >>> data_loader = raschka.gpt.create_dataloader_v1(text_data, **kwargs)
    >>> model = raschka.gpt.GPTModel(GPT_CONFIG)
    >>> loss = raschka.gpt_train.calc_loss_loader(data_loader, model, device, num_batches=2)
    """)
    dependent_members = ('raschka.gpt.create_dataloader_v1', 'raschka.gpt.GPTModel', 'torch.utils.data.DataLoader',
                         'raschka.gpt_train.calc_loss_batch')
    return docstring, dependent_members

def check_raschka_gpt_generate_text_simple():
    docstring = textwrap.dedent("""\
    >>> cfg = dict(emb_dim=6, context_length=4, n_heads=3, drop_rate=0.5, qkv_bias=False, vocab_size=6, n_layers=2)
    >>> model = raschka.gpt.GPTModel(cfg).eval()
    >>> encoded_tensor = torch.randint(0, cfg['vocab_size'], (2, cfg["context_length"]))
    >>> out = raschka.gpt.generate_text_simple(
    ...     model=model,
    ...     idx=encoded_tensor,
    ...     max_new_tokens=10,
    ...     context_size=cfg["context_length"]
    ... )
    """)
    dependent_members = ('raschka.gpt.GPTModel',)
    return docstring, dependent_members

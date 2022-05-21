import torch as th


def get_model(name='bert-base-chinese'):
    return th.hub.load('huggingface/pytorch-transformers', 'model', name)


def get_tokenizer(name='bert-base-chinese'):
    return th.hub.load('huggingface/pytorch-transformers', 'tokenizer', name)


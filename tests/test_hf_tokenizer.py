import torch

from open_clip.tokenizer import HFTokenizer


class StubTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    cls_token_id = 3
    pad_token_id = 0
    sep_token_id = 2

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": [[10, 11, 12], [20]]}
        return {"input_ids": torch.tensor([[1, 10, 2, 0], [1, 20, 2, 0]])}


def make_hf_tokenizer(mode=""):
    tokenizer = object.__new__(HFTokenizer)
    tokenizer.tokenizer = StubTokenizer()
    tokenizer.tokenizer_mode = mode
    tokenizer.context_length = 4
    tokenizer.clean_fn = lambda text: text.strip()
    tokenizer.strip_sep_token = False
    return tokenizer


def test_hf_tokenizer_uses_callable_interface():
    tokenizer = make_hf_tokenizer()

    result = tokenizer([" first ", "second"])

    assert torch.equal(result, torch.tensor([[1, 10, 2, 0], [1, 20, 2, 0]]))
    assert tokenizer.tokenizer.calls == [
        (["first", "second"], {
            "return_tensors": "pt",
            "max_length": 4,
            "padding": "max_length",
            "truncation": True,
        })
    ]


def test_hf_tokenizer_clips_mode_uses_callable_interface():
    tokenizer = make_hf_tokenizer(mode="clips")

    result = tokenizer(["first", "second"])

    assert torch.equal(result, torch.tensor([[1, 10, 2, 3], [1, 20, 2, 3]]))
    assert tokenizer.tokenizer.calls == [
        (["first", "second"], {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "return_tensors": None,
        })
    ]

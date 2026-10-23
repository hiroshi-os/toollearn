"""Shared serialization/device choices for training and generation."""
import random


def seed_all(seed):
    from transformers import set_seed
    random.seed(seed)
    set_seed(seed)


def render(tokenizer, messages, chat_template=False, thinking=None):
    if not chat_template:
        if thinking is not None:
            raise ValueError("thinking requires a supported chat template")
        from .experiment import prompt_text
        return prompt_text(messages)
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template")
    kwargs = {}
    if thinking is not None:
        if "enable_thinking" not in tokenizer.chat_template:
            raise ValueError("template does not explicitly support enable_thinking")
        kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, **kwargs)


def load_model(model_name, revision=None):
    import torch
    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=dtype)
    return model.to(device)

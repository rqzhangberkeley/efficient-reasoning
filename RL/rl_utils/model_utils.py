from transformers import AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig

from ..configs import GRPOConfig, SFTConfig


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    tokenizer.padding_side = 'left'
    # RZ: This is a workaround to avoid a bug in the Qwen2 tokenizer.
    # RZ: The bug is that the tokenizer uses the right padding side by default.
    # RZ: This causes problems with the forward method of the model.
    # RZ: The bug is fixed in the latest version of the Qwen2 tokenizer.
    # RZ: But we are using a custom tokenizer here.
    # RZ: So we need to set the padding side to the left.
    

    return tokenizer

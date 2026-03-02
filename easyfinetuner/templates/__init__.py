"""
Prompt templates for easyfinetuner.
"""

from .alpaca import (
    ALPACA_TEMPLATE,
    ALPACA_TEMPLATE_NO_INPUT,
    ALPACA_SHORT,
    apply_alpaca_template,
    EXAMPLE_FORMATS,
)

from .chatml import (
    CHATML_TEMPLATE,
    CHATML_WITH_SYSTEM,
    apply_chatml_template,
    convert_messages_to_chatml,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    SYSTEM_PROMPTS,
)

__all__ = [
    "ALPACA_TEMPLATE",
    "ALPACA_TEMPLATE_NO_INPUT",
    "ALPACA_SHORT",
    "apply_alpaca_template",
    "EXAMPLE_FORMATS",
    "CHATML_TEMPLATE",
    "CHATML_WITH_SYSTEM",
    "apply_chatml_template",
    "convert_messages_to_chatml",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "ROLE_ASSISTANT",
    "SYSTEM_PROMPTS",
]

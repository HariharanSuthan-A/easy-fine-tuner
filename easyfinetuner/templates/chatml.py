"""
ChatML-style prompt templates for chat/conversation datasets.
ChatML is used by models like Qwen, Nous-Hermes, and others.
"""

# Standard ChatML template for single turn
CHATML_TEMPLATE = """<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|><|endoftext|>"""

# ChatML with system message
CHATML_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|><|endoftext|>"""

# Multi-turn conversation template
CHATML_MULTI_TURN = """{conversation}<|endoftext|>"""

# Short version
CHATML_SHORT = """<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


def format_message(role: str, content: str) -> str:
    """Format a single message in ChatML format."""
    return f"<|im_start|>{role}\n{content}<|im_end|>"


def apply_chatml_template(
    input_text: str,
    output_text: str,
    system: str = "",
    history: list = None
) -> str:
    """
    Apply ChatML template to format chat data.
    
    Args:
        input_text: User input
        output_text: Assistant output
        system: Optional system message
        history: Optional list of previous messages as dicts with 'role' and 'content'
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    # System message
    if system:
        parts.append(format_message("system", system))
    
    # History
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(format_message(role, content))
    
    # Current turn
    parts.append(format_message("user", input_text))
    parts.append(format_message("assistant", output_text))
    
    return "\n".join(parts) + "<|endoftext|>"


def convert_messages_to_chatml(messages: list) -> str:
    """
    Convert a list of messages to ChatML format.
    
    Args:
        messages: List of dicts with 'role' and 'content' keys
        
    Returns:
        Formatted conversation string
    """
    parts = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(format_message(role, content))
    
    return "\n".join(parts) + "<|endoftext|>"


# Role constants
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Common system prompts
SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "helpful": "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible.",
    "code": "You are a coding assistant. Provide clear, efficient, and well-commented code solutions.",
    "creative": "You are a creative writing assistant. Help with stories, poems, and creative content.",
    "analytical": "You are an analytical assistant. Provide detailed analysis and reasoning.",
    "concise": "You are a concise assistant. Provide brief, direct answers without unnecessary elaboration.",
}

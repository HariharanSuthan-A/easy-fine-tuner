"""
Alpaca-style prompt templates for instruction-following datasets.
"""

# Standard Alpaca template with input field
ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<|endoftext|>"""

# Alpaca template without input field
ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}<|endoftext|>"""

# Short version for concise outputs
ALPACA_SHORT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<|endoftext|>"""

# Template with system message
ALPACA_WITH_SYSTEM = """{system}

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<|endoftext|>"""

# Helper function to apply template
def apply_alpaca_template(
    instruction: str,
    output: str,
    input_text: str = "",
    system: str = "",
    use_input: bool = True
) -> str:
    """
    Apply Alpaca template to format instruction data.
    
    Args:
        instruction: The task instruction
        output: The expected output/response
        input_text: Optional additional context
        system: Optional system message
        use_input: Whether to include input field
        
    Returns:
        Formatted prompt string
    """
    if system:
        return ALPACA_WITH_SYSTEM.format(
            system=system,
            instruction=instruction,
            input=input_text,
            output=output
        )
    
    if use_input and input_text:
        return ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        return ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output
        )


# Example formats for common use cases
EXAMPLE_FORMATS = {
    "qa": {
        "instruction": "Answer the following question.",
        "input": "{question}",
        "output": "{answer}"
    },
    "summarization": {
        "instruction": "Summarize the following text.",
        "input": "{text}",
        "output": "{summary}"
    },
    "classification": {
        "instruction": "Classify the following text into one of the categories.",
        "input": "{text}",
        "output": "{label}"
    },
    "extraction": {
        "instruction": "Extract the requested information from the text.",
        "input": "{text}",
        "output": "{extracted}"
    },
    "generation": {
        "instruction": "{task_description}",
        "input": "{context}",
        "output": "{generated}"
    }
}

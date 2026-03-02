"""
Example 2: Advanced Fine-tuning with Custom Configuration

This notebook demonstrates advanced features including custom config,
different data formats, evaluation, and more.
"""

# %% [markdown]
# # Example 2: Advanced Fine-tuning 🎯
# 
# This notebook explores advanced features of EasyFinetuner:
# - Custom configuration
# - Different data formats
# - Multiple evaluation metrics
# - Saving/loading adapters

# %%
from easyfinetuner import FineTuner, DataProcessor
import json

# %% [markdown]
# ## 1. Create a Larger Dataset
# 
# Let's create a more substantial dataset for better results:

# %%
training_data = []

# General knowledge
knowledge_qa = [
    {"instruction": "Explain quantum computing in simple terms.", "input": "", "output": "Quantum computing uses quantum mechanics principles like superposition and entanglement to perform calculations. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can be both 0 and 1 simultaneously, enabling them to solve certain problems much faster."},
    {"instruction": "What is machine learning?", "input": "", "output": "Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed. It includes techniques like neural networks, decision trees, and statistical models."},
    {"instruction": "How does blockchain work?", "input": "", "output": "Blockchain is a distributed ledger technology where transactions are recorded in blocks, linked together cryptographically. Each block contains a hash of the previous block, creating an immutable chain. It's decentralized, meaning no single entity controls it."},
    {"instruction": "What is climate change?", "input": "", "output": "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the industrial revolution, human activities—primarily burning fossil fuels—have increased greenhouse gas concentrations, causing Earth's average temperature to rise."},
    {"instruction": "Explain DNA.", "input": "", "output": "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for all known living organisms. It's a double helix structure made of nucleotides (A, T, C, G) that encode the information needed to build and maintain an organism."},
]
training_data.extend(knowledge_qa)

# Creative writing
creative = [
    {"instruction": "Write a haiku about nature.", "input": "", "output": "Whispers through the trees\nAutumn leaves dance gracefully\nNature's symphony"},
    {"instruction": "Continue this story: The old lighthouse keeper woke up to an unusual sound...", "input": "", "output": "...coming from the tower above. He grabbed his lantern and climbed the spiral stairs, heart pounding. At the top, he found a baby seagull trapped in the weather vane. Gently freeing it, he smiled as it soared into the dawn sky, circling back as if to say thank you before disappearing into the golden sunrise."},
    {"instruction": "Describe a cyberpunk city.", "input": "", "output": "Neon lights pierce through perpetual rain in the sprawling metropolis. Towering skyscrapers display holographic advertisements while flying cars navigate between buildings. In the shadows below, augmented humans with neural implants trade information in hidden data markets, their cybernetic eyes glowing with digital overlays of the grim reality around them."},
]
training_data.extend(creative)

# Code
programming = [
    {"instruction": "Write a Python function to reverse a string.", "input": "", "output": "def reverse_string(s):\n    return s[::-1]\n\n# Example usage\nprint(reverse_string('hello'))  # Output: olleh"},
    {"instruction": "Explain list comprehensions in Python.", "input": "", "output": "List comprehensions provide a concise way to create lists in Python. Syntax: [expression for item in iterable if condition].\n\nExample:\nsquares = [x**2 for x in range(10) if x % 2 == 0]\n# Result: [0, 4, 16, 36, 64]"},
    {"instruction": "What is the difference between a list and a tuple?", "input": "", "output": "Lists are mutable (can be modified after creation) and use square brackets []. Tuples are immutable (cannot be changed after creation) and use parentheses (). Tuples are slightly faster and can be used as dictionary keys, while lists cannot."},
]
training_data.extend(programming)

# Mathematical
math_qa = [
    {"instruction": "What is the Pythagorean theorem?", "input": "", "output": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c²"},
    {"instruction": "Explain the Fibonacci sequence.", "input": "", "output": "The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21... It appears in many biological settings and has connections to the golden ratio (approximately 1.618)."},
    {"instruction": "What is a derivative in calculus?", "input": "", "output": "A derivative measures the instantaneous rate of change of a function at a point. Geometrically, it's the slope of the tangent line to the function's graph at that point. Denoted as f'(x) or df/dx."},
]
training_data.extend(math_qa)

print(f"Total training examples: {len(training_data)}")

# Save to JSON for demonstration
with open('training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)

# %% [markdown]
# ## 2. Analyze Dataset
# 
# Let's see what EasyFinetuner detects about our data:

# %%
dataset = DataProcessor.load_dataset('training_data.json')
stats = DataProcessor.analyze_dataset(dataset)

print("Dataset Statistics:")
print(f"  Examples: {stats['num_examples']}")
print(f"  Format: {stats['format_type']}")
print(f"  Avg length: {stats['avg_length']} tokens")
print(f"  Max length: {stats['max_length']} tokens")
print(f"  P95 length: {stats['p95_length']} tokens")

# %% [markdown]
# ## 3. Initialize with Custom Configuration
# 
# Instead of auto-config, we'll specify exact parameters:

# %%
tuner = FineTuner(
    model_name="unsloth/Qwen3-4B",  # Larger model for better quality
    max_seq_length=1024,
    load_in_4bit=True,
    template="alpaca",  # Explicitly use Alpaca template
)

# %% [markdown]
# ## 4. Train with Full Control

# %%
print("Starting training with custom configuration...\n")

stats = tuner.train(
    dataset='training_data.json',
    output_dir="outputs/advanced_model",
    num_epochs=5,
    learning_rate=1e-4,
    batch_size=2,
    lora_r=64,  # Higher rank for better adaptation
    lora_alpha=128,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    weight_decay=0.01,
    validation_split=0.1,
    seed=42,
)

print(f"\n✓ Training Complete!")
print(f"  Final loss: {stats['final_loss']:.4f}")
print(f"  Runtime: {stats['train_runtime_formatted']}")
print(f"  Samples/sec: {stats['samples_per_second']:.2f}")

# %% [markdown]
# ## 5. Comprehensive Evaluation

# %%
print("Running comprehensive evaluation...\n")

metrics = tuner.evaluate(
    metrics=["perplexity"],
    num_samples=20,
    generate_report=True,
    output_path="outputs/evaluation_report.html"
)

print("\nEvaluation Results:")
for metric, value in metrics.items():
    if metric != 'report_path':
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

print(f"\nReport saved to: {metrics.get('report_path')}")

# %% [markdown]
# ## 6. Test Various Prompts

# %%
test_prompts = [
    ("Explain artificial intelligence.", 200),
    ("Write a short poem about technology.", 150),
    ("What is the quadratic formula?", 100),
    ("Describe a futuristic city.", 200),
]

print("\nTesting the fine-tuned model:\n")
for prompt, max_tokens in test_prompts:
    response = tuner.predict(prompt, max_new_tokens=max_tokens, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
    print("-" * 60)

# %% [markdown]
# ## 7. Save and Reload
# 
# Demonstrate saving adapters and loading them later:

# %%
# Save only the LoRA adapters (small, ~10-50MB)
tuner.save("outputs/lora_adapters", save_adapter_only=True)
print("✓ LoRA adapters saved")

# Save full merged model (larger, ~4-8GB)
# tuner.save("outputs/full_model", save_adapter_only=False)
# print("✓ Full model saved")

# %% [markdown]
# ## 8. Export Multiple Quantizations

# %%
quantizations = ["q4_k_m", "q5_k_m"]

for quant in quantizations:
    print(f"\nExporting {quant}...")
    path = tuner.export_gguf(
        output_path=f"outputs/model_{quant}.gguf",
        quantization=quant
    )
    print(f"  Saved: {path}")

# %% [markdown]
# ## 🎉 Advanced Example Complete!
# 
# You've learned:
# - Custom training configuration
# - Dataset analysis
# - Multiple evaluation metrics
# - Saving/loading adapters
# - Multiple GGUF exports
# 
# Next: Try Example 3 for chatbot training!

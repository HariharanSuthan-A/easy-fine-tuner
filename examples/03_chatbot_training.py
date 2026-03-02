"""
Example 3: Building a Chatbot with EasyFinetuner

This notebook demonstrates training a conversational AI using
chat format data with the ChatML template.
"""

# %% [markdown]
# # Example 3: Chatbot Training 💬
# 
# This notebook shows how to train a chatbot using conversational data
# with the ChatML format (used by Qwen, Nous-Hermes, and other models).

# %%
from easyfinetuner import FineTuner, DataProcessor
import json

# %% [markdown]
# ## 1. Create Chat Training Data
# 
# Chat format uses a list of messages with roles (system, user, assistant):

# %%
chat_data = [
    {
        "messages": [
            {"role": "user", "content": "Hi! Can you help me with a coding problem?"},
            {"role": "assistant", "content": "Hello! I'd be happy to help you with your coding problem. What language are you working with, and what issue are you facing?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do I reverse a list in Python?"},
            {"role": "assistant", "content": "You can reverse a list in Python using several methods:\n\n1. Using the reverse() method (modifies in place):\n   my_list.reverse()\n\n2. Using slicing (creates new list):\n   reversed_list = my_list[::-1]\n\n3. Using reversed() function:\n   reversed_list = list(reversed(my_list))\n\nChoose based on whether you need to modify the original or create a new list!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What's the difference between a list and a tuple?"},
            {"role": "assistant", "content": "Great question! Here are the key differences:\n\n**Lists:**\n- Mutable (can be modified)\n- Use square brackets: []\n- Slower\n- Good for collections that change\n\n**Tuples:**\n- Immutable (cannot be modified after creation)\n- Use parentheses: ()\n- Faster and use less memory\n- Can be dictionary keys\n- Good for fixed collections\n\nUse lists when data changes, tuples when it shouldn't!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a branch of AI where computers learn from data without being explicitly programmed. Instead of following hard-coded rules, ML algorithms identify patterns in training data and use them to make predictions or decisions on new data.\n\nCommon types include:\n- Supervised learning (learning from labeled examples)\n- Unsupervised learning (finding patterns in unlabeled data)\n- Reinforcement learning (learning through trial and error)\n\nApplications include image recognition, language translation, recommendation systems, and more!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Can you explain neural networks in simple terms?"},
            {"role": "assistant", "content": "Think of a neural network like a virtual brain made of layers of connected 'neurons':\n\n1. **Input Layer**: Receives information (like pixels in an image)\n2. **Hidden Layers**: Process and transform the data through weighted connections\n3. **Output Layer**: Produces the result (like 'this is a cat')\n\nEach connection has a 'weight' that gets adjusted during training. The network learns by:\n- Making predictions\n- Checking how wrong it was\n- Adjusting weights to be more accurate\n\nIt's inspired by biological brains but much simpler - yet powerful enough to recognize speech, drive cars, and play games!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Write me a short poem about coding"},
            {"role": "assistant", "content": "Lines of logic, pure and bright,\nDancing through the silent night,\nFunctions call and loops repeat,\nMaking the impossible complete.\n\nSemicolons, brackets tight,\nBugs appear, then vanish from sight,\nIn this digital poets' frame,\nWe give the void a name."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do I handle errors in Python?"},
            {"role": "assistant", "content": "Python uses try-except blocks for error handling:\n\n```python\ntry:\n    # Code that might cause an error\n    result = 10 / 0\nexcept ZeroDivisionError:\n    # Handle specific error\n    print(\"Can't divide by zero!\")\nexcept Exception as e:\n    # Handle any other error\n    print(f\"An error occurred: {e}\")\nelse:\n    # Runs if no exception\n    print(\"Success!\")\nfinally:\n    # Always runs (cleanup)\n    print(\"Done\")\n```\n\nBest practices:\n- Catch specific exceptions, not generic ones\n- Use finally for cleanup\n- Log errors for debugging\n- Don't silently ignore errors"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What are some good Python libraries for data science?"},
            {"role": "assistant", "content": "Essential Python libraries for data science:\n\n**Core:**\n- NumPy: Numerical computing, arrays\n- Pandas: Data manipulation and analysis\n- Matplotlib/Seaborn: Data visualization\n\n**Machine Learning:**\n- Scikit-learn: Classical ML algorithms\n- TensorFlow/PyTorch: Deep learning\n- XGBoost/LightGBM: Gradient boosting\n\n**Data Processing:**\n- Requests/BeautifulSoup: Web scraping\n- SQLAlchemy: Database interaction\n- Dask: Parallel computing\n\n**Jupyter:**\n- Jupyter Notebook/Lab: Interactive development\n- IPython: Enhanced interactive shell\n\nStart with NumPy and Pandas - they're the foundation!"}
        ]
    },
]

# Add more conversations by duplicating with variations
for i in range(3):  # Triplicate for more data
    chat_data.extend(chat_data)

print(f"Total conversation examples: {len(chat_data)}")

# Save to JSONL (one conversation per line)
with open('chat_data.jsonl', 'w') as f:
    for conv in chat_data:
        f.write(json.dumps(conv) + '\n')

# %% [markdown]
# ## 2. Verify Data Format

# %%
# Check the first example
with open('chat_data.jsonl', 'r') as f:
    first_line = json.loads(f.readline())
    print("Sample conversation:")
    print(json.dumps(first_line, indent=2))

# Analyze the dataset
dataset = DataProcessor.load_dataset('chat_data.jsonl')
stats = DataProcessor.analyze_dataset(dataset)

print(f"\nDataset stats:")
print(f"  Format detected: {stats['format_type']}")
print(f"  Examples: {stats['num_examples']}")
print(f"  Avg length: {stats['avg_length']} tokens")

# %% [markdown]
# ## 3. Initialize Chat Model
# 
# Use ChatML template for conversational models:

# %%
tuner = FineTuner(
    model_name="unsloth/Qwen3-4B",
    max_seq_length=2048,  # Longer for conversations
    load_in_4bit=True,
    template="chatml",  # ChatML format
)

# %% [markdown]
# ## 4. Train the Chatbot

# %%
print("Training chatbot...\n")

stats = tuner.train(
    dataset='chat_data.jsonl',
    output_dir="outputs/chatbot",
    num_epochs=3,
    learning_rate=2e-4,
    batch_size=1,
    gradient_accumulation_steps=8,
    lora_r=32,
    validation_split=0.05,  # Small validation for chat
    warmup_steps=20,
)

print(f"\n✓ Chatbot training complete!")
print(f"  Final loss: {stats['final_loss']:.4f}")
print(f"  Time: {stats['train_runtime_formatted']}")

# %% [markdown]
# ## 5. Interactive Chat Session

# %%
def chat_session(tuner, system_prompt=None):
    """Run an interactive chat session."""
    print("="*60)
    print("🤖 Chatbot Ready! Type 'quit' to exit.")
    print("="*60)
    
    conversation_history = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        # Build prompt with conversation history
        if conversation_history:
            prompt = ""
            for msg in conversation_history:
                if msg["role"] == "system":
                    prompt += f"<|im_start|>system\n{msg['content']} "
                elif msg["role"] == "user":
                    prompt += f"<|im_start|>user\n{msg['content']} "
                elif msg["role"] == "assistant":
                    prompt += f"<|im_start|>assistant\n{msg['content']} "
            prompt += f"<|im_start|>user\n{user_input} "
            prompt += "<|im_start|>assistant\n"
        else:
            prompt = user_input
        
        # Generate response
        response = tuner.predict(
            prompt,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Update history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Keep only last 6 exchanges to prevent context overflow
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]
        
        print(f"\nBot: {response}")

# Run interactive session (uncomment to use)
# chat_session(tuner)

# %% [markdown]
# ## 6. Test with Predefined Conversations

# %%
test_conversations = [
    "Explain object-oriented programming",
    "What's the best way to learn coding?",
    "Help me debug: my Python script keeps crashing",
    "Tell me a joke about programmers",
    "What's the difference between AI and ML?",
]

print("Testing chatbot responses:\n")
for user_msg in test_conversations:
    response = tuner.predict(
        f"<|im_start|>user\n{user_msg} <|im_start|>assistant\n",
        max_new_tokens=200,
        temperature=0.7
    )
    print(f"User: {user_msg}")
    print(f"Bot: {response}\n")
    print("-" * 60)

# %% [markdown]
# ## 7. Export for Different Platforms

# %%
# Export for Ollama/llama.cpp
print("Exporting for local inference...")

quantizations = {
    "q4_k_m": "Fast, good quality (~1.5GB)",
    "q5_k_m": "Slower, better quality (~2GB)",
}

for quant, desc in quantizations.items():
    path = tuner.export_gguf(
        output_path=f"outputs/chatbot_{quant}.gguf",
        quantization=quant
    )
    print(f"✓ {quant}: {path}")
    print(f"  ({desc})")

# %% [markdown]
# ## 8. Create Ollama Modelfile

# %%
modelfile_content = '''FROM ./chatbot_q4_k_m.gguf

TEMPLATE """<|im_start|>system
You are a helpful AI assistant trained to assist with coding questions, explain technical concepts, and engage in friendly conversation. Be concise but thorough in your explanations. 
<|im_start|>user
{{ .Prompt }} 
<|im_start|>assistant
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
'''

with open('outputs/Modelfile', 'w') as f:
    f.write(modelfile_content)

print("\n✓ Ollama Modelfile created at outputs/Modelfile")
print("\nTo use with Ollama:")
print("  1. Copy the .gguf file and Modelfile to your Ollama models directory")
print("  2. Run: ollama create mychatbot -f Modelfile")
print("  3. Chat: ollama run mychatbot")

# %% [markdown]
# ## 🎉 Chatbot Training Complete!
# 
# You've created a conversational AI that can:
# - Answer coding questions
# - Explain technical concepts
# - Engage in conversation
# - Run locally on your machine
# 
# ### Tips for better chatbots:
# - Use more diverse training data (1000+ conversations)
# - Include system prompts for personality
# - Fine-tune for longer (5-10 epochs)
# - Use larger models (8B+) for better responses
# - Add safety guidelines in system prompts

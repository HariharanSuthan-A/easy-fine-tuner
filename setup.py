"""
Setup script for easyfinetuner package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="easyfinetuner",
    version="0.1.0",
    author="Hariharan Suthan",
    author_email="your.email@example.com",
    description="Dead simple LLM fine-tuning with Unsloth",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HariharanSuthan-A/easy-fine-tuner",
    project_urls={
        "Bug Tracker": "https://github.com/HariharanSuthan-A/easy-fine-tuner/issues",
        "Documentation": "https://github.com/HariharanSuthan-A/easy-fine-tuner#readme",
        "Source": "https://github.com/HariharanSuthan-A/easy-fine-tuner",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords="llm fine-tuning unsloth lora peft transformers ai machine-learning",
    install_requires=requirements,
    extras_require={
        "eval": [
            "sacrebleu",
            "rouge-score",
            "nltk",
        ],
        "viz": [
            "matplotlib",
            "seaborn",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "easyfinetuner=easyfinetuner.cli:main",
        ],
    },
)

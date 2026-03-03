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
    author="EasyFinetuner Team",
    author_email="",
    description="Dead simple LLM fine-tuning with Unsloth",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/easyfinetuner",
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

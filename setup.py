from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="aeroattention",
    version="1.0.0",  
    author="MeforgersAI",
    author_email="aixr@meforgers.com",
    description="AeroAttention: An Advanced Quantum-Enhanced Attention Mechanism for Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeforgersDev/AeroAttention",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19.5",
        "torch>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
        "cuda": [
            "triton>=2.1.0",
            "flash-attn>=2.5.6"
        ],
    },
    include_package_data=True,
    license="Apache License 2.0",
    keywords="attention mechanism transformer quantum machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/MeforgersDev/AeroAttention/issues",
        "Source": "https://github.com/MeforgersDev/AeroAttention",
    },
)
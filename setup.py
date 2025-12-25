from setuptools import setup, find_packages

setup(
    name="unswag",
    version="0.3.0",
    author="John Augustine Young",
    description="UnSwag v0.3: Packet-Switched Attention & 1-Bit Structural Isomorphism",
    long_description="""
    # UnSwag v0.3: The Evolution of Attention

    UnSwag is a high-performance library designed to break the memory wall and compute ceiling. 
    Version 0.3.0 introduces **Protocol C**, moving beyond simple memory optimization into 
    intelligent, hardware-native semantic routing.

    ## Version 0.3.0 Highlights (Protocol C):
    - **Hardware-Native Semantic Routing:** 2-bit packet switching (00, 01, 10, 11) for differential token processing.
    - **Causal Hybrid Path:** Integrated Depthwise-Separable CNNs for syntactic speedups (25x theoretical) 
      alongside Sparse Global Attention.
    - **Adaptive Summary Register:** Recursive O(1) memory register for abstract sequence gist.
    
    ## Legacy Foundations (v0.1 - v0.2):
    - **1-Bit Structural Isomorphism:** 32x activation memory reduction via ReLU mapping.
    - **TPU/GPU Native:** Support for JAX/Pallas (TPU) and OpenAI Triton (GPU).
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/augstentatious/unswagai",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "jax",
        "jaxlib",
        "triton",
        "numpy",
        "einops",
        "librosa",
        "torchaudio",
        "transformers"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)

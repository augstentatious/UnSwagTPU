from setuptools import setup, find_packages

setup(
    name="sophia_pallas",
    version="0.1.0",
    description="Structural Isomorphism & PEFT Library for JAX/TPU",
    author="John Augustine Young", # The Principal Investigator
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "transformers"
    ],
)


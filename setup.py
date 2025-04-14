from setuptools import setup, find_packages

setup(
    name="diffusion-factor-model",
    version="0.1.0",
    description="Diffusion Factor Models: Generating High-Dimensional Financial Data with Factor Structure",
    author="xuym",
    author_email="xuyumin@pku.edu.cn",
    url="https://github.com/xuym/diffusion-factor-model",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "einops",
        "tqdm",
        "matplotlib",
        "ema-pytorch>=0.2.0",
        "accelerate>=0.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "seaborn>=0.11.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
)

"""Setup script for MTL-GNN-DTA package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="mtl-gnn-dta",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Task Learning Graph Neural Network for Drug-Target Affinity Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MTL-GNN-DTA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.22.0"
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinxcontrib-napoleon>=0.7"
        ]
    },
    entry_points={
        "console_scripts": [
            "mtl-gnn-dta=mtl_gnn_dta.cli.main:main",
            "mtl-gnn-dta-train=mtl_gnn_dta.cli.train:main",
            "mtl-gnn-dta-predict=mtl_gnn_dta.cli.predict:main",
            "mtl-gnn-dta-preprocess=mtl_gnn_dta.cli.preprocess:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mtl_gnn_dta": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
)
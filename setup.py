"""
Setup script for ehrsequencing package.
Alternative to pyproject.toml for environments with SSL issues.
"""

from setuptools import setup, find_packages

setup(
    name="ehrsequencing",
    version="0.1.0",
    description="Biological language model for Electronic Health Records",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        # All dependencies installed via conda environment.yml
        # No pip dependencies needed
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ]
    },
)

"""Setup script for the Amharic E-commerce Data Extractor."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="amharic-ecommerce-extractor",
    version="1.0.0",
    description="Extract and process Amharic e-commerce data from Telegram channels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EthioMart Data Team",
    author_email="data@ethiomart.com",
    url="https://github.com/ethiomart/amharic-ecommerce-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ethiomart-task1=run_task1:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ethiomart/amharic-ecommerce-extractor/issues",
        "Source": "https://github.com/ethiomart/amharic-ecommerce-extractor",
    },
) 
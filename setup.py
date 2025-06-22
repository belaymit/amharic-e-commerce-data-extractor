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
    description="Extract and process Amharic e-commerce data from Telegram channels using NER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amharic E-commerce Data Extractor Contributors",
    author_email="contributors@amharic-extractor.dev",
    url="https://github.com/your-username/amharic-ecommerce-data-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Amharic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "amharic-extractor=main:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
        ],
        "visualization": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "plotly>=5.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/amharic-ecommerce-data-extractor/issues",
        "Source": "https://github.com/your-username/amharic-ecommerce-data-extractor",
        "Documentation": "https://github.com/your-username/amharic-ecommerce-data-extractor/blob/main/README.md",
    },
    keywords="amharic nlp ner ecommerce telegram machine-learning transformers",
    include_package_data=True,
    zip_safe=False,
) 
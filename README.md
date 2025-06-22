<a name="readme-top"></a>

<div align="center">

  <br/>

  <h3><b>Amharic E-commerce Data Extractor</b></h3>

</div>

<!-- TABLE OF CONTENTS -->

# 📗 Table of Contents

- [📖 About the Project](#about-project)
  - [🛠 Built With](#built-with)
    - [Tech Stack](#tech-stack)
    - [Key Features](#key-features)
- [💻 Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Install](#install)
  - [Usage](#usage)
  - [Run tests](#run-tests)
  - [Deployment](#deployment)
- [👥 Authors](#authors)
- [🔭 Future Features](#future-features)
- [🤝 Contributing](#contributing)
- [⭐️ Show your support](#support)
- [🙏 Acknowledgements](#acknowledgements)
- [❓ FAQ (OPTIONAL)](#faq)
- [📝 License](#license)

<!-- PROJECT DESCRIPTION -->

# 📖 Amharic E-commerce Data Extractor <a name="about-project"></a>

**Amharic E-commerce Data Extractor** is an advanced NLP project that leverages transformer models to extract products, prices, and locations from Amharic Telegram e-commerce channels. The system supports FinTech applications by providing vendor scoring capabilities for micro-lending decisions.

## 🛠 Built With <a name="built-with"></a>

### Tech Stack <a name="tech-stack"></a>

<details>
  <summary>Machine Learning</summary>
  <ul>
    <li><a href="https://pytorch.org/">PyTorch</a></li>
    <li><a href="https://huggingface.co/transformers/">Transformers</a></li>
    <li><a href="https://scikit-learn.org/">Scikit-learn</a></li>
  </ul>
</details>

<details>
  <summary>Data Processing</summary>
  <ul>
    <li><a href="https://www.python.org/">Python</a></li>
    <li><a href="https://pandas.pydata.org/">Pandas</a></li>
    <li><a href="https://numpy.org/">NumPy</a></li>
  </ul>
</details>

<details>
<summary>Data Collection</summary>
  <ul>
    <li><a href="https://docs.telethon.dev/">Telethon</a></li>
    <li><a href="https://docs.pyrogram.org/">Pyrogram</a></li>
  </ul>
</details>

<details>
<summary>Interpretability</summary>
  <ul>
    <li><a href="https://shap.readthedocs.io/">SHAP</a></li>
    <li><a href="https://lime-ml.readthedocs.io/">LIME</a></li>
  </ul>
</details>

<!-- Features -->

### Key Features <a name="key-features"></a>

- **Named Entity Recognition for Amharic text** - Extract products, prices, and locations
- **Multi-model comparison framework** - XLM-Roberta, DistilBERT, and mBERT  
- **Model interpretability with SHAP and LIME** - Transparent AI decision making
- **Telegram data collection pipeline** - Automated scraping from Ethiopian channels
- **FinTech vendor scorecard** - Complete micro-lending risk assessment system
- **Business intelligence dashboard** - Vendor analytics and lending recommendations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## 💻 Getting Started <a name="getting-started"></a>

To get a local copy up and running, follow these steps.

### Prerequisites

In order to run this project you need:

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup

Clone this repository to your desired folder:

```sh
cd my-folder
git clone https://github.com/your-username/amharic-ecommerce-data-extractor.git
cd amharic-ecommerce-data-extractor
```

### Install

Install this project with:

```sh
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python setup_project.py
```

### Usage

To run the project, execute the following commands:

```sh
# Quick demo (4-5 seconds)
python scripts/demo_task3_training.py

# Data collection
python scripts/run_task1.py

# Interactive notebooks
cd notebooks/
jupyter notebook
```

**Available Notebooks:**
- `NER_Fine_Tuning.ipynb` - Train NER models
- `Model_Comparison.ipynb` - Compare model performance  
- `Model_Interpretability.ipynb` - Analyze model decisions
- `CoNLL_Labeling.ipynb` - Data labeling and annotation
- `Vendor_Scorecard.ipynb` - FinTech lending analysis

### Run tests

To run tests, run the following command:

```sh
python -m pytest tests/
```

### Deployment

You can deploy this project using:

```sh
# Build Docker container
docker build -t amharic-ner .

# Run with Docker Compose
docker-compose up
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- AUTHORS -->

## 👥 Authors <a name="authors"></a>

👤 **Belay Birhanu G.**

- LinkedIn: [@LinkedIn](https://www.linkedin.com/in/belay-bgwa/)
- GitHub: [GitHub](https://github.com/belaymit)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- FUTURE FEATURES -->

## 🔭 Future Features <a name="future-features"></a>

- [ ] **Real-time entity extraction API**
- [ ] **Multi-language support (Oromo, Tigrinya)**
- [ ] **Advanced FinTech risk scoring models**
- [ ] **Web interface for model interaction**
- [ ] **Automated model retraining pipeline**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## 🤝 Contributing <a name="contributing"></a>

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](../../issues/).

**Development Process:**
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`python -m pytest`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SUPPORT -->

## ⭐️ Show your support <a name="support"></a>

If you find this project helpful for Ethiopian e-commerce or Amharic NLP research, please give it a star! Your support helps improve Amharic language processing tools.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGEMENTS -->

## 🙏 Acknowledgments <a name="acknowledgements"></a>

- **Ethiopian Telegram E-commerce Community** for providing rich data sources
- **Hugging Face** for transformer models and datasets library
- **KAIM (Kigali AI & ML)** for project guidance and structure
- **Microverse** for the README template
- **Open source NLP community** for tools and libraries

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- FAQ (optional) -->

## ❓ FAQ <a name="faq"></a>

- **What entities can this system extract?**
  - Products (ቦርሳ, ጫማ, cream, iPhone), Prices (ዋጋ 5000 ብር), Locations (ቦሌ, መርካቶ)

- **Which models perform best for Amharic NER?**
  - XLM-Roberta generally performs best, but DistilBERT offers good speed-accuracy balance

- **How accurate is the system?**
  - Target F1-score >0.85 with proper training (3+ epochs on full dataset)

- **Can I use this for other Ethiopian languages?**
  - Currently optimized for Amharic, but framework can be adapted for Oromo, Tigrinya

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## 📝 License <a name="license"></a>

This project is [MIT](./MIT.md) licensed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

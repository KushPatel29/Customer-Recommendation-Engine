# Sales Data Analysis and Customer Recommendation Pipeline

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [Running the Pipeline](#running-the-pipeline)
- [Functionality](#functionality)
  - [1. Web Scraping](#1-web-scraping)
  - [2. Data Analysis](#2-data-analysis)
  - [3. Text-Based Similarity](#3-text-based-similarity)
  - [4. Main Pipeline](#4-main-pipeline)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This Python project automates the process of analyzing sales data to identify top customers across different regions, scrape additional information about these customers from Bing, recommend new potential customers, and generate actionable leads. The pipeline culminates in comprehensive reports in both Excel and CSV formats, facilitating strategic decision-making for sales and marketing teams.

## Features

- **Top Customer Identification**: Determines the top N customers per region based on revenue.
- **Web Scraping**: Extracts business information and similar businesses for each top customer using Bing search.
- **Data Analysis**: Recommends new customers not in the top N but with significant revenue contributions.
- **Text-Based Similarity**: (Optional) Computes similarity scores between top customers based on scraped text data.
- **Comprehensive Reporting**: Generates multi-sheet Excel reports and consolidated CSV files for easy data consumption.

## Demo

![Pipeline Workflow](https://github.com/yourusername/sales-data-pipeline/blob/main/demo/workflow.png)

*Illustration of the pipeline workflow.*

## Prerequisites

- **Python**: Version 3.7 or higher
- **Internet Connection**: Required for web scraping
- **Microsoft Excel**: (Optional) For viewing Excel reports

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/sales-data-pipeline.git
   cd sales-data-pipeline
   ```

2. **Create a Virtual Environment** (Optional but recommended)

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

3. **Install Required Python Packages**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the necessary packages manually:*

   ```bash
   pip install requests beautifulsoup4 numpy pandas nltk tensorflow scikit-learn openpyxl
   ```

## Setup

1. **NLTK Stopwords**

   The script utilizes NLTK's stopwords for text preprocessing. Ensure you download the necessary NLTK data.

   ```python
   import nltk
   nltk.download("stopwords")
   ```

   *Alternatively, run the following command in your terminal:*

   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

2. **Prepare Sales Data**

   Ensure you have your sales data in an Excel file (e.g., `Sales_Data.xlsx`) with at least the following columns:

   - `RegionName`
   - `CustomerName`
   - `Rev` (Revenue)
   - `OrderId`
   - `QuantityOrdered`

   Place the Excel file in an accessible directory and note its path for the next step.

## Usage

### Running the Pipeline

1. **Configure Script Parameters**

   In the `sales_pipeline.py` script, locate the `if __name__ == "__main__":` block at the end of the script. Update the following parameters as needed:

   - `excel_path`: Path to your sales data Excel file.
   - `top_n`: Number of top customers per region to analyze (default is 50).
   - `min_lineitem_revenue`: Minimum revenue threshold for recommending new customers (default is 500).
   - `output_file`: Desired name for the output CSV file (default is `final_output.csv`).

   ```python
   if __name__ == "__main__":
       # 1) Load your Excel file
       excel_path = r"C:\Path\To\Your\Sales_Data.xlsx"
       sales_data = pd.read_excel(excel_path, sheet_name=0)

       # 2) Run the pipeline
       run_final_pipeline(
           data=sales_data,
           top_n=50,
           min_lineitem_revenue=500,
           output_file="final_output.csv"
       )
   ```

2. **Execute the Script**

   Run the script using Python:

   ```bash
   python sales_pipeline.py
   ```

   *Replace `sales_pipeline.py` with the actual name of your Python script file.*

3. **Review the Outputs**

   Upon successful execution, the script generates:

   - **Excel Report** (`final_output.xlsx`):
     - **Top50**: Top N customers per region.
     - **Recommended**: New customer recommendations.
     - **NewLeads**: Leads from similar businesses.
     - **Similarity** (Optional): Cosine similarity matrix of top customers.
   - **CSV Report** (`final_output.csv`):
     - Consolidated data with a `Type` column indicating the category (`Top50`, `Recommended`, `NewLead`).

## Functionality

### 1. Web Scraping

- **`scrape_customer_details(customer_name)`**

  Scrapes Bing to retrieve basic business information about a given customer. Returns the top 5 titles and snippets from the search results.

- **`scrape_similar_businesses_to_customer(customer_name, region, limit=5)`**

  Searches for similar businesses to the specified customer within a region on Bing. Returns a list of dictionaries containing titles and snippets of the top results.

### 2. Data Analysis

- **`get_top_n_customers_all_regions(data, n=50)`**

  Aggregates revenue data to identify the top N customers per region.

- **`recommend_new_customers(data, top_n_df, min_lineitem_revenue=500)`**

  Recommends new customers who are not in the top N but have line-item revenues exceeding the specified threshold.

### 3. Text-Based Similarity

- **`preprocess_text(text_list)`**

  Cleans and preprocesses text data by removing punctuation and stopwords.

- **`build_bow_similarity(top_customers_scraped)`**

  Constructs a Bag-of-Words (BOW) model from the scraped text data and computes a cosine similarity matrix between top customers.

### 4. Main Pipeline

- **`run_final_pipeline(data, top_n=50, min_lineitem_revenue=500, output_file="final_output.csv")`**

  Orchestrates the entire workflow:

  1. **Identify Top N Customers**: Determines the top N customers per region based on revenue.
  2. **Scrape Customer Details**: Retrieves additional information for each top customer from Bing.
  3. **Build Text-Based Similarity**: (Optional) Computes similarity scores between top customers based on scraped text data.
  4. **Recommend New Customers**: Identifies potential new customers based on revenue criteria.
  5. **Scrape Similar Businesses**: Gathers leads from similar businesses related to top customers.
  6. **Generate Reports**: Outputs the results to both Excel and CSV formats.

## Output

1. **Excel File (`final_output.xlsx`)**

   - **Top50**: Lists the top N customers per region with their total revenue and scraped text data.
   - **Recommended**: Contains recommended new customers with revenue, order frequency, and quantity metrics.
   - **NewLeads**: Details leads obtained from scraping similar businesses to top customers.
   - **Similarity**: (If enabled) Shows the cosine similarity scores between top customers based on their scraped text.

2. **CSV File (`final_output.csv`)**

   Consolidated data combining Top50, Recommended, and NewLeads with a `Type` column to distinguish between different categories.

## Troubleshooting

- **Missing NLTK Data**

  If you encounter errors related to missing NLTK data (e.g., `stopwords`), ensure you've downloaded the necessary data using:

  ```python
  import nltk
  nltk.download("stopwords")
  ```

- **HTTP Errors During Scraping**

  - **Status Code Issues**: If Bing returns non-200 status codes, verify your internet connection and ensure Bing is accessible.
  - **IP Blocking**: Excessive scraping may lead to IP blocking. Implement rate limiting or use proxies if necessary.

- **Incorrect File Paths**

  Ensure that the `excel_path` in the main block correctly points to your sales data Excel file.

- **Missing Columns in Sales Data**

  The script expects specific columns (`RegionName`, `CustomerName`, `Rev`, `OrderId`, `QuantityOrdered`). Verify that your Excel file contains these columns.

- **TensorFlow Issues**

  Ensure that TensorFlow is properly installed and compatible with your Python version. If you encounter installation issues, refer to the [TensorFlow Installation Guide](https://www.tensorflow.org/install).

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

   Describe your changes and submit for review.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing.
- [NLTK](https://www.nltk.org/) for natural language processing.
- [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) for text tokenization.
- [Scikit-learn](https://scikit-learn.org/) for machine learning utilities.
- [Pandas](https://pandas.pydata.org/) for data manipulation.
- [NumPy](https://numpy.org/) for numerical operations.
- [Requests](https://docs.python-requests.org/en/latest/) for HTTP requests.

---

*For any questions or support, please contact [your.email@example.com](mailto:your.email@example.com).*

# Quick Start

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/sales-data-pipeline.git
   cd sales-data-pipeline
   ```

2. **Set Up the Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords')"
   ```

3. **Prepare Your Sales Data**

   Ensure your Excel file (`Sales_Data.xlsx`) is formatted correctly with the required columns.

4. **Run the Pipeline**

   ```bash
   python sales_pipeline.py
   ```

   Replace `sales_pipeline.py` with your script's filename if different.

5. **Review Outputs**

   Check the generated `final_output.xlsx` and `final_output.csv` for your analysis results.


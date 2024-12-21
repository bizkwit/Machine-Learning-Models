# Machine-Learning-Models

[![GitHub license](https://img.shields.io/github/license/bizkwit/Machine-Learning-Models)](https://github.com/bizkwit/Machine-Learning-Models/blob/main/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/bizkwit/Machine-Learning-Models/graphs/commit-activity)


## Overview

This repository contains implementations of several machine learning models, including linear regression, k-nearest neighbors, and support vector machines.  The models are demonstrated using the Wisconsin Breast Cancer dataset and Google stock price data (for linear regression).  Both scikit-learn implementations and custom algorithms are provided for k-nearest neighbors, allowing for a comparison of performance and understanding of the underlying mechanics.


## Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Examples](#examples)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)


## Features

- **Linear Regression:** Predicts Google stock prices using historical data. Includes visualization of predictions.  Both a scikit-learn implementation (`lr.py`) and a custom implementation (`lr_algo.py`) are provided.
- **K-Nearest Neighbors:** Classifies breast cancer data. Includes a scikit-learn implementation (`k-nearest.py`) and a custom algorithm (`knearest_algo.py`) for comparison.
- **Support Vector Machine (SVM):** Classifies breast cancer data using an SVM classifier (`svm.py`).
- **K-Means Clustering:** (Code present but not fully utilized in `kmean.py`)  This section could be expanded upon.


## Technologies

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Quandl](https://www.quandl.com/) (for financial data in linear regression example)


## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages:  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `quandl`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bizkwit/Machine-Learning-Models.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Machine-Learning-Models
   ```
3. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt 
   ```
   *(Note: A `requirements.txt` file should be created listing the necessary packages.)*

### Usage

Each Python file (`lr.py`, `k-nearest.py`, `knearest_algo.py`, `svm.py`, `kmean.py`, `lr_algo.py`) contains a complete example of a specific model. Run them individually using:
```bash
python3 <filename>.py
```
Remember to have the `breast-cancer-wisconsin.data` file in the same directory.  The Quandl API key is also needed for the linear regression example (`lr.py`).


## Examples

See the individual Python files for detailed examples of each model's usage and implementation.  The `lr.py` file, for instance, demonstrates how to use linear regression to predict stock prices and visualize the results.  `knearest_algo.py` shows a step-by-step implementation of the k-nearest neighbors algorithm.


## Roadmap

- [x] Implement Linear Regression (Scikit-learn & Custom)
- [x] Implement K-Nearest Neighbors (Scikit-learn & Custom)
- [x] Implement Support Vector Machine
- [x] Include K-Means Clustering (Expand upon existing code)
- [ ] Add more datasets
- [ ] Implement model evaluation metrics (precision, recall, F1-score etc.)
- [ ] Add unit tests


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## License

[Specify your license here, e.g., MIT License]

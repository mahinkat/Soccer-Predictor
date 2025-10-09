# Premier League Predictor

## Overview
Premier League Predictor is a machine learning-powered tool designed to predict Premier League standings using historical match data. It analyzes 5 seasons of match statistics and uses Random Forest classification to forecast league outcomes, including championship predictions, European qualification spots, and relegation candidates.

## Features
* Predicts Premier League standings based on historical data
* Machine learning model with rolling averages for team performance
* Generates predictions for:
  * League champion
  * Top 4 teams (Champions League qualification)
  * Europa League qualifiers
  * Relegation candidates
* Normalizes statistics to a standard 38-game season

## Tech Stack
* **Language**: Python 3.x
* **Libraries**: pandas, scikit-learn, numpy
* **Machine Learning**: Random Forest Classifier
* **Data**: Premier League match statistics (2020-2025)

## Installation

### Prerequisites
* Python (>= 3.8)
* pip

### Steps
1. Clone the repository:
```bash
git clone https://github.com/mahinkat/Soccer-Predictor.git
cd Soccer-Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the predictor:
```bash
python premier.py
```

The script will process the data and output predicted league standings.


## How It Works
1. **Data Processing**: Combines and processes 5 seasons of Premier League match data
2. **Feature Engineering**: Creates rolling averages for goals, shots, fouls, and cards
3. **Model Training**: Trains a Random Forest classifier on matches before 2023
4. **Prediction**: Predicts outcomes for matches after 2023
5. **Simulation**: Simulates full league standings based on predictions

## Model Details
* **Algorithm**: Random Forest Classifier
* **Features**: 
  * Venue (Home/Away)

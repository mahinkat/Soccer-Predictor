# Soccer Predictor

## Overview
Soccer Predictor is a machine learning-powered tool designed to predict European Top 5 League (Serie-A, Ligue1, Premier League, Bundesliga, LaLiga) standings using historical match data. It analyzes 5 seasons of match statistics and uses Random Forest classification to forecast league outcomes, including Champions League predictions, Europa League qualification spots, and relegation candidates.

## Features
* Predicts European Top 5 League standings based on historical data
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
* **Data**: European Top 5 League match statistics (2020-2025)

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
pip3 install -r requirements.txt
```

3. Run the universal predictor for the Top 5 Leagues :
```bash
python3 main.py
```
4. OR Run each individual league predictor:
```bash
python3 bundesliga.py
python3 laliga.py
python3 ligue1.py
python3 premier.py
python3 seriea.py
```

The script will process the data and output predicted league standings.

## How It Works
1. **Data Processing**: Combines and processes 5 seasons of match data for each league
2. **Feature Engineering**: Creates rolling averages for goals, shots, fouls, and cards
3. **Model Training**: Trains a Random Forest classifier on matches before 2023
4. **Prediction**: Predicts outcomes for matches after 2023
5. **Simulation**: Simulates full league standings based on predictions

## Model Details
* **Algorithm**: Random Forest Classifier
* **Features**: 
  * Venue (Home/Away)
  * Opponent strength
  * Day of week
  * Rolling averages (goals for/against, shots, shots on target)
* **Parameters**: 40 estimators, min_samples_split=300
* **Training/Test Split**: Pre-2023 / Post-2023

## Data Format
CSV files should contain the following columns:
* Date, HomeTeam, AwayTeam
* FTHG, FTAG (Full Time Home/Away Goals)
* HS, AS, HST, AST (Shots and Shots on Target)
* HF, AF (Fouls)
* HY, AY, HR, AR (Yellow and Red Cards)
* Referee

## Contribution Guidelines
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make changes and commit: `git commit -m "Added new feature"`.
4. Push the branch: `git push origin feature-branch`.
5. Open a Pull Request.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, feel free to reach out via [GitHub](https://github.com/mahinkat).

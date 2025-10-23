import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import cross_val_score

warnings.simplefilter(action='ignore', category=FutureWarning)

# File mapping
FILE_MAP = {
    "serieascsv/serieaseason-2425.csv": 2025,
    "serieascsv/serieaseason-2324.csv": 2024,
    "serieascsv/serieaseason-2223.csv": 2023,
    "serieascsv/serieaseason-2122.csv": 2022,
    "serieascsv/serieaseason-2021.csv": 2021
}

FINAL_COLS = [
    'date', 'day', 'comp', 'venue', 'result', 'gf', 'ga', 'opponent',
    'referee', 'sh', 'sot', 'foul_comm', 'foul_rec', 'yc_rec', 'rc_rec',
    'yc_opp', 'rc_opp', 'season', 'team', 'day_code'
]

# Result function
def get_result(ftr, venue):
    if venue == 'Home':
        return 'W' if ftr=='H' else 'L' if ftr=='A' else 'D'
    elif venue == 'Away':
        return 'W' if ftr=='A' else 'L' if ftr=='H' else 'D'
    return None

# Process season CSV
def process_season_data(file_name, season_year):
    try:
        df = pd.read_csv(file_name)
        df['date'] = pd.to_datetime(df['Date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d')
        df['season'] = season_year

        home_df = pd.DataFrame({
            'date': df['date'], 'team': df['HomeTeam'], 'opponent': df['AwayTeam'], 'venue': 'Home',
            'gf': df['FTHG'], 'ga': df['FTAG'], 'result': df.apply(lambda row: get_result(row['FTR'], 'Home'), axis=1),
            'sh': df['HS'], 'sot': df['HST'], 'referee': df['Referee'],
            'season': df['season'], 'foul_comm': df['HF'], 'foul_rec': df['AF'],
            'yc_rec': df['HY'], 'rc_rec': df['HR'], 'yc_opp': df['AY'], 'rc_opp': df['AR']
        })

        away_df = pd.DataFrame({
            'date': df['date'], 'team': df['AwayTeam'], 'opponent': df['HomeTeam'], 'venue': 'Away',
            'gf': df['FTAG'], 'ga': df['FTHG'], 'result': df.apply(lambda row: get_result(row['FTR'], 'Away'), axis=1),
            'sh': df['AS'], 'sot': df['AST'], 'referee': df['Referee'],
            'season': df['season'], 'foul_comm': df['AF'], 'foul_rec': df['HF'],
            'yc_rec': df['AY'], 'rc_rec': df['AR'], 'yc_opp': df['HY'], 'rc_opp': df['HR']
        })

        combined = pd.concat([home_df, away_df], ignore_index=True)
        combined['comp'] = 'Serie A'
        combined['day'] = pd.to_datetime(combined['date']).dt.day_name().str[:3]
        combined['day_code'] = pd.to_datetime(combined['date']).dt.dayofweek

        for col in ['gf','ga','sh','sot','foul_comm','foul_rec','yc_rec','rc_rec','yc_opp','rc_opp']:
            combined[col] = combined[col].astype(float)

        return combined[FINAL_COLS]
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return pd.DataFrame(columns=FINAL_COLS)

# Combine all seasons
all_data = [process_season_data(f, y) for f, y in FILE_MAP.items()]
combined_data = pd.concat(all_data, ignore_index=True)
combined_data.sort_values(by=['date','team'], inplace=True, ignore_index=True)

# Encode categorical features
combined_data["venue_code"] = combined_data["venue"].astype("category").cat.codes
combined_data["opp_code"] = combined_data["opponent"].astype("category").cat.codes
combined_data["target"] = (combined_data["result"]=="W").astype(int)
predictors = ["venue_code", "opp_code", "day_code"]

# Compute rolling averages
def rolling_averages(group, cols, new_cols, window=3):
    group = group.sort_values("date")
    group[new_cols] = group[cols].rolling(window, closed='left').mean()
    return group.dropna(subset=new_cols)

cols = ["gf","ga","sh","sot"]
new_cols = [f"{c}_rolling" for c in cols]
matches_rolling = combined_data.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols)).droplevel(0)
matches_rolling.index = range(matches_rolling.shape[0])

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1),
    "XGBoost": XGBClassifier(n_estimators=25, max_depth=1, eval_metric='logloss'),
    "Logistic Regression": LogisticRegression(max_iter=250)
}

# Cross-validation
def cross_val_precision(model, data, predictors, cv=5):
    scorer = make_scorer(precision_score, zero_division=0)
    scores = cross_val_score(model, data[predictors], data['target'], cv=cv, scoring=scorer)
    return scores.mean(), scores.std()

# Select best model
best_model_name = None
best_score = -1
for name, model in models.items():
    mean, std = cross_val_precision(model, matches_rolling, predictors+new_cols)
    print(f"{name}: Precision = {mean:.3f} ± {std:.3f}")
    if mean > best_score:
        best_score = mean
        best_model_name = name

print(f"\n✅ Best model based on CV precision: {best_model_name}")

# Train best model
train = matches_rolling[matches_rolling["date"]<'2023-01-01']
test = matches_rolling[matches_rolling["date"]>'2023-01-01']
best_model = models[best_model_name]
best_model.fit(train[predictors+new_cols], train['target'])
preds = best_model.predict(test[predictors+new_cols])

# Prepare merged dataframe for simulation
combined_merged = pd.DataFrame({
    'team_x': test['team'],
    'opponent_x': test['opponent'],
    'predicted_x': preds
})

# Simulation parameters
POINTS_FOR_WIN = 3
POINTS_FOR_DRAW = 1
GAMES_PER_SERIE_A_SEASON = 38
DRAW_PROBABILITY_ON_ZERO = 0.25

def simulate_league(df):
    league_table = {team:{'Points':0,'Played':0,'Wins':0,'Draws':0,'Losses':0} 
                    for team in pd.unique(df['team_x'].tolist()+df['opponent_x'].tolist())}
    
    for _, row in df.iterrows():
        home, away, pred = row['team_x'], row['opponent_x'], int(row['predicted_x'])
        league_table[home]['Played'] += 1
        league_table[away]['Played'] += 1
        if pred == 1:
            league_table[home]['Points'] += POINTS_FOR_WIN
            league_table[home]['Wins'] += 1
            league_table[away]['Losses'] += 1
        else:
            if np.random.rand() < DRAW_PROBABILITY_ON_ZERO:
                league_table[home]['Draws'] += 1
                league_table[home]['Points'] += POINTS_FOR_DRAW
                league_table[away]['Draws'] += 1
                league_table[away]['Points'] += POINTS_FOR_DRAW
            else:
                league_table[home]['Losses'] += 1
                league_table[away]['Wins'] += 1
                league_table[away]['Points'] += POINTS_FOR_WIN

    table = pd.DataFrame.from_dict(league_table, orient='index')
    table = table[table['Played']>0]
    norm_factor = table['Played']/GAMES_PER_SERIE_A_SEASON
    for col in ['Wins','Draws','Losses','Points']:
        table[f'Avg. {col}'] = table[col]/norm_factor
    table['Avg. Played'] = GAMES_PER_SERIE_A_SEASON
    final_cols = ['Avg. Played','Avg. Wins','Avg. Draws','Avg. Losses','Avg. Points']
    table = table[final_cols].sort_values(by=['Avg. Points','Avg. Wins'], ascending=[False,False])
    return table

# Run simulation
final_table = simulate_league(combined_merged)
num_seasons_data = round(final_table['Avg. Played'].max()/GAMES_PER_SERIE_A_SEASON)
if num_seasons_data<1: num_seasons_data=1
final_table_display = final_table.round(2)

# Print results
print("\n" + "="*70)
print(f"  PREDICTED SERIE A TABLE ({best_model_name}, Averaged over Data Span of {num_seasons_data} Seasons)")
print("="*70)
print(final_table_display.to_string())
print("="*70)

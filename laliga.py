import pandas as pd
import platform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("laligascsv/laligaseason-2425.csv")
home_teams = df['HomeTeam'].unique()
away_teams = df['AwayTeam'].unique()
teams = pd.unique(pd.concat([pd.Series(home_teams), pd.Series(away_teams)]))


FILE_MAP = {
    "laligascsv/laligaseason-2425.csv": 2025,
    "laligascsv/laligaseason-2324.csv": 2024,
    "laligascsv/laligaseason-2223.csv": 2023,
    "laligascsv/laligaseason-2122.csv": 2022,
    "laligascsv/laligaseason-2021.csv": 2021
}

FINAL_COLS = [
    'date', 'day', 'comp', 'venue', 'result', 'gf', 'ga', 'opponent',
    'referee', 'sh', 'sot', 'foul_comm', 'foul_rec', 'yc_rec', 'rc_rec',
    'yc_opp', 'rc_opp', 'season', 'team', 'day_code'
]

def get_result(ftr, venue):
    if venue == 'Home':
        if ftr == 'H': return 'W'
        elif ftr == 'A': return 'L'
        else: return 'D'
    elif venue == 'Away':
        if ftr == 'A': return 'W'
        elif ftr == 'H': return 'L'
        else: return 'D'
    return None

def process_season_data(file_name, season_year):
    try:
        season_df = pd.read_csv(file_name)

        season_df['date'] = pd.to_datetime(season_df['Date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d')
        season_df['season'] = season_year

        home_df = pd.DataFrame({
            'date': season_df['date'],
            'team': season_df['HomeTeam'],
            'opponent': season_df['AwayTeam'],
            'venue': 'Home',
            'gf': season_df['FTHG'],
            'ga': season_df['FTAG'],
            'result': season_df.apply(lambda row: get_result(row['FTR'], 'Home'), axis=1),
            'sh': season_df['HS'],
            'sot': season_df['HST'],
            'referee': season_df['Referee'],
            'season': season_df['season'],
            'foul_comm': season_df['HF'],
            'foul_rec': season_df['AF'],
            'yc_rec': season_df['HY'],
            'rc_rec': season_df['HR'],
            'yc_opp': season_df['AY'],
            'rc_opp': season_df['AR']
        })

        away_df = pd.DataFrame({
            'date': season_df['date'],
            'team': season_df['AwayTeam'],
            'opponent': season_df['HomeTeam'],
            'venue': 'Away',
            'gf': season_df['FTAG'],
            'ga': season_df['FTHG'],
            'result': season_df.apply(lambda row: get_result(row['FTR'], 'Away'), axis=1),
            'sh': season_df['AS'],
            'sot': season_df['AST'],
            'referee': season_df['Referee'],
            'season': season_df['season'],
            'foul_comm': season_df['AF'],
            'foul_rec': season_df['HF'],
            'yc_rec': season_df['AY'],
            'rc_rec': season_df['AR'],
            'yc_opp': season_df['HY'],
            'rc_opp': season_df['HR']
        })

        combined_df = pd.concat([home_df, away_df], ignore_index=True)

        combined_df['comp'] = 'LaLiga'
        combined_df['date_dt'] = pd.to_datetime(combined_df['date'])
        combined_df['day'] = combined_df['date_dt'].dt.day_name().str[:3]
        combined_df['day_code'] = combined_df['date_dt'].dt.dayofweek
        combined_df = combined_df.drop(columns=['date_dt'])

        float_cols = [
            'gf', 'ga', 'sh', 'sot', 'foul_comm', 'foul_rec', 'yc_rec', 'rc_rec', 'yc_opp', 'rc_opp'
        ]
        for col in float_cols:
            combined_df[col] = combined_df[col].astype(float)

        final_df = combined_df[FINAL_COLS].copy()

        return final_df
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return pd.DataFrame(columns=FINAL_COLS)

all_seasons_data = []

for file_name, season_year in FILE_MAP.items():
    processed_df = process_season_data(file_name, season_year)
    all_seasons_data.append(processed_df)

combined_data = pd.concat(all_seasons_data, ignore_index=True)

combined_data.sort_values(by=['date', 'team'], inplace=True, ignore_index=True)

combined_file_name = 'laligascsv/laligaall_seasons_combined_sorted.csv'
combined_data.to_csv(combined_file_name, index=False)

combined_data["venue_code"] = combined_data["venue"].astype("category").cat.codes
combined_data["opp_code"] = combined_data["opponent"].astype("category").cat.codes
combined_data["day_code"] = pd.to_datetime(combined_data["date"]).dt.dayofweek
combined_data["target"] = (combined_data["result"] == "W").astype("int")

rf = RandomForestClassifier(n_estimators = 40, min_samples_split = 300, random_state=1)
train = combined_data[combined_data["date"] < '2023-01-01']
test = combined_data[combined_data["date"] > '2023-01-01']
predictors = ["venue_code", "opp_code", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])
acc = accuracy_score(test["target"], preds)
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])
precision_score(test["target"], preds)

grouped_matches = combined_data.groupby("team")
group = grouped_matches.get_group("Barcelona")
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group
cols = ["gf","ga","sh","sot"]
new_cols = [f"{c}_rolling" for c in cols]
rolling_averages(group,cols,new_cols)
matches_rolling = combined_data.groupby("team").apply(lambda x: rolling_averages(x,cols,new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

def make_predictions(data,predictors):
    train = data[data["date"] < '2023-01-01']
    test = data[data["date"] > '2023-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual = test["target"], predicted = preds), index = test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["date","team","opponent", "result"]], left_index=True, right_index=True)
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton": "Brighton"}
mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()
merged.to_csv("laligascsv/laligamerged.csv",index=False)


POINTS_FOR_WIN = 3
POINTS_FOR_DRAW = 1
POINTS_FOR_LOSS = 0
GAMES_PER_LALIGA_SEASON = 38 
TOTAL_FIXTURES_PER_SEASON = 380 

DRAW_PROBABILITY_ON_ZERO = 0.25

def simulate_league_cumulative(df):
    print(f"--- Running Cumulative Simulation based on {len(df)} Fixtures ---")

    home_teams = df['team_x'].unique()
    away_teams = df['opponent_x'].unique()
    all_teams = sorted(list(set(list(home_teams) + list(away_teams))))

    league_table = {}
    for team in all_teams:
        league_table[team] = {
            'Points': 0,
            'Played': 0,
            'Wins': 0,
            'Draws': 0,
            'Losses': 0,        }

    for index, row in df.iterrows():
        home_team = row['team_x']
        away_team = row['opponent_x']
        prediction = int(row['predicted_x']) 

        league_table[home_team]['Played'] += 1
        league_table[away_team]['Played'] += 1

        if prediction == 1:
            league_table[home_team]['Points'] += POINTS_FOR_WIN
            league_table[home_team]['Wins'] += 1
            league_table[away_team]['Losses'] += 1

        elif prediction == 0:
           
            if np.random.rand() < DRAW_PROBABILITY_ON_ZERO:
                league_table[home_team]['Draws'] += 1
                league_table[home_team]['Points'] += POINTS_FOR_DRAW
                
                league_table[away_team]['Draws'] += 1
                league_table[away_team]['Points'] += POINTS_FOR_DRAW
            else:
                league_table[home_team]['Losses'] += 1
                
                league_table[away_team]['Points'] += POINTS_FOR_WIN
                league_table[away_team]['Wins'] += 1


    final_table = pd.DataFrame.from_dict(league_table, orient='index')
    final_table = final_table[final_table['Played'] > 0]
    
    return final_table


try:
    df_predictions = pd.read_csv('laligascsv/laligamerged.csv')

    df_predictions.columns = df_predictions.columns.str.strip().str.lower()
    
    required_cols_lower = ['team_x', 'opponent_x', 'predicted_x']
    if not all(col in df_predictions.columns for col in required_cols_lower):
        raise ValueError(f"CSV file must contain columns: {required_cols_lower}")

    cumulative_table = simulate_league_cumulative(df_predictions)

    if cumulative_table.empty:
        print("\nNo teams found in the prediction data.")
        raise SystemExit(0) 
        
    averaged_table = cumulative_table.copy()

    normalization_factor = averaged_table['Played'] / GAMES_PER_LALIGA_SEASON
    
    print(f"\nNOTE: Normalizing each team's statistics to an average single {GAMES_PER_LALIGA_SEASON}-match season.")

    cols_to_normalize = ['Wins', 'Draws', 'Losses', 'Points']
    
    for col in cols_to_normalize:
        new_col_name = f'Avg. {col}'
        averaged_table[new_col_name] = averaged_table[col].div(normalization_factor).fillna(0) 

    averaged_table['Avg. Played'] = float(GAMES_PER_LALIGA_SEASON)
    final_cols = ['Avg. Played', 'Avg. Wins', 'Avg. Draws', 'Avg. Losses', 'Avg. Points']
    final_table = averaged_table[final_cols].copy()

    final_table = final_table.sort_values(
        by=['Avg. Points', 'Avg. Wins'], 
        ascending=[False, False]
    )

    final_table_display = final_table.round(2)
    
    max_played = cumulative_table['Played'].max()
    num_seasons_data = round(max_played / GAMES_PER_LALIGA_SEASON)
    if num_seasons_data < 1: num_seasons_data = 1

    print("\n" + "="*70)
    print(f"  PREDICTED LALIGA TABLE (Averaged over Data Span of {num_seasons_data} Seasons)")
    print("="*70)
    print(final_table_display.to_string())
    print("="*70)

    if not final_table.empty:
        print(f"\n--- Average Predicted Standings ---\n")
        avg_champ = final_table.index[0]
        avg_points = final_table_display['Avg. Points'].iloc[0] 
        print(f"Predicted Champion (Average Season): {avg_champ} with {avg_points} points.")
    
        if len(final_table) >= 20:
             relegation_teams = final_table.index[-3:].tolist()
             print(f"Predicted Relegation (Average Season): {', '.join(relegation_teams)}.")

        champions_league = final_table.index[:4].tolist()
        print(f"Predicted UEFA Champions League Group Stage (Average Season): {', '.join(champions_league)}.")

        europa_league= final_table.index[5]
        europa_league2 = final_table.index[6]
        print(f"Predicted Europa League Group Stage (Average Season): {europa_league}, {europa_league2}")
        

except FileNotFoundError:
    print(f"Error: The file 'laligamerged.csv' was not found.")
    print("Please make sure 'laligamerged.csv' is in the same directory as this script.")
except ValueError as e:
    print(f"Data Error: {e}")
except SystemExit:
    pass
except Exception as e:
    print(f"An unexpected error occurred: {e}")

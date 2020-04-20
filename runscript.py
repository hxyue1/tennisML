import tennisML as tML
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import os

def demo():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    ###This runscript gives an example of how you could go about making predictions for the 2020 Australian Open given data from 2000-2019.

    #Reading in data
    deuce_atp = pd.read_csv('Data/atp_matches_deuce_demo.csv')

    #Cleaning data and transforming player statistics
    deuce_atp_cleaned = tML.data_cleaning(deuce_atp, 'Hard', ['Grand Slams', '250 or 500', 'Davis Cup', 'Masters', 'Challenger', 'Tour Finals'], 2000)
    deuce_atp_long = tML.convert_long(deuce_atp_cleaned)
    deuce_atp_long = tML.engineer_player_stats(deuce_atp_long)

    #Aggregating statistics
    last_cols = ['player_rank', 'player_log_rank', 'player_rank_points', 'player_log_rank_points', 'player_ht', 'player_old_elo']
    rolling_cols = ['player_serve_win_ratio', 'player_return_win_ratio', 'player_bp_per_game', 'player_bp_conversion_ratio', 'player_game_win_ratio', 'player_point_win_ratio', 'player_clutch_factor', 'player_win_rank_weight', 'player_win_elo_weight', 'player_game_win_ratio_rank_weighted', 'player_point_win_ratio_rank_weighted', 'player_game_win_ratio_elo_weighted', 'player_point_win_ratio_elo_weighted']
    tournaments = ['Australian Open','US Open', 'Us Open']
    ma_features, _ = tML.aggregate_features(deuce_atp_long, tournaments, 2000, rolling_cols, last_cols, 60, 365)


    wrangled_df = tML.wrangle_target(deuce_atp_cleaned, 2000, tournaments)
    merged_data = tML.merge_data(wrangled_df, ma_features, 2000, tournaments, True)
    df_final = tML.get_player_difference(merged_data)

    ML_cols_subset= [
    'player_1_win', 'player_rank_diff', 'player_log_rank_diff',
    'player_rank_points_diff', 'player_log_rank_points_diff',
    'player_serve_win_ratio_diff', 'player_return_win_ratio_diff',
    'player_bp_per_game_diff',
    'player_game_win_ratio_diff', 'player_point_win_ratio_diff',
    'player_clutch_factor_diff', 
    'player_win_rank_weight_diff', 'player_win_elo_weight_diff',
    'player_point_win_ratio_rank_weighted_diff',
    'player_point_win_ratio_elo_weighted_diff', 'player_old_elo_diff',
        'player_game_win_ratio_rank_weighted_diff', 'player_game_win_ratio_elo_weighted_diff'
        
    ]

    #Using XGBoost, but you can use any classifier you want, change the code accordingly.
    model = XGBClassifier(
        objective = "binary:logistic"
    )

    features = ML_cols_subset.copy()
    features.remove('player_1_win')
    train_indices, val_indices, test_indices = tML.f_chain_index_by_year(2016,2019, 20, True)
    test_scores = []
    val_scores = []

    #Training, validating and testing our model
    for train_index, val_index, test_index in zip(train_indices, val_indices, test_indices):

        print(train_index, val_index)

        X_train = df_final.loc[df_final.tourney_start_date.dt.year.isin(train_index), features]
        y_train = df_final.loc[df_final.tourney_start_date.dt.year.isin(train_index), 'player_1_win']
        X_val = df_final.loc[df_final.tourney_start_date.dt.year == val_index, features]
        y_val = df_final.loc[df_final.tourney_start_date.dt.year == val_index, 'player_1_win']
        X_test = df_final.loc[(df_final.tourney_start_date.dt.year == test_index)&\
                            (df_final.tourney_name.isin(tournaments)), features]
        y_test = df_final.loc[(df_final.tourney_start_date.dt.year == test_index)&\
                            (df_final.tourney_name.isin(tournaments)), 'player_1_win']


        model.fit(X_train, 
                y_train, 
                eval_set = [(X_val, y_val)],
                eval_metric="logloss", 
                early_stopping_rounds = 20) 

        val_score = min(model.evals_result()['validation_0']['logloss'])
        val_scores.append(val_score)

        test_preds = model.predict_proba(X_test)
        test_score = log_loss(y_test, test_preds[:,1])
        test_scores.append(test_score) 
        
    print(np.mean(test_scores))

    players = ['Rafael Nadal', 'Hugo Dellien', 'Federico Delbonis', 'Joao Sousa', 'Christopher Eubanks', 'Peter Gojowczyk', 'Jozef Kovalik', 'Pablo Carreno Busta', 'Nick Kyrgios', 'Lorenzo Sonego', 'Pablo Cuevas', 'Gilles Simon', 'Yasutaka Uchiyama', 'Mikael Ymer', 'Mario Vilella Martinez', 'Karen Khachanov', 'Gael Monfils', 'Yen hsun Lu', 'Ivo Karlovic', 'Vasek Pospisil', 'James Duckworth', 'Aljaz Bedene', 'Ernests Gulbis', 'Felix Auger Aliassime', 'Taylor Harry Fritz', 'Tallon Griekspoor', 'Ilya Ivashka', 'Kevin Anderson', 'Alex Bolt', 'Albert Ramos', 'Adrian Mannarino', 'Dominic Thiem', 'Daniil Medvedev', 'Francis Tiafoe', 'Dominik Koepfer', 'Pedro Martinez', 'Hugo Gaston', 'Jaume Munar', 'Alexei Popyrin', 'Jo Wilfried Tsonga', 'John Isner', 'Thiago Monteiro', 'Alejandro Tabilo', 'Daniel Elahi Galan', 'Miomir Kecmanovic', 'Andreas Seppi', 'Damir Dzumhur', 'Stan Wawrinka', 'David Goffin', 'Jeremy Chardy', 'Pierre Hugues Herbert', 'Cameron Norrie', 'Yuichi Sugita', 'Elliot Benchetrit', "Christopher OConnell", 'Andrey Rublev', 'Nikoloz Basilashvili', 'Soon woo Kwon', 'Fernando Verdasco', 'Evgeny Donskoy', 'Casper Ruud', 'Egor Gerasimov', 'Marco Cecchinato', 'Alexander Zverev', 'Matteo Berrettini', 'Andrew Harris', 'Tennys Sandgren', 'Marco Trungelliti', 'Roberto Carballes Baena', 'Ricardas Berankis', 'Sam Querrey', 'Borna Coric', 'Guido Pella', 'John Patrick Smith', 'Mohamed Safwat', 'Gregoire Barrere', 'Jordan Thompson', 'Alexander Bublik', 'Reilly Opelka', 'Fabio Fognini', 'Denis Shapovalov', 'Marton Fucsovics', 'Jannik Sinner', 'Max Purcell', 'Leonardo Mayer', 'Tommy Paul', 'Juan Ignacio Londero', 'Grigor Dimitrov', 'Hubert Hurkacz', 'Dennis Novak', 'John Millman', 'Ugo Humbert', 'Quentin Halys', 'Filip Krajinovic', 'Steve Johnson', 'Roger Federer', 'Stefanos Tsitsipas', 'Salvatore Caruso', 'Philipp Kohlschreiber', 'Marcos Giron', 'Christian Garin', 'Stefano Travaglia', 'Lorenzo Giustino', 'Milos Raonic', 'Benoit Paire', 'Cedrik Marcel Stebe', 'Marin Cilic', 'Corentin Moutet', 'Pablo Andujar', 'Michael Mmoh', 'Feliciano Lopez', 'Roberto Bautista Agut', 'Diego Sebastian Schwartzman', 'Lloyd George Muirhead Harris', 'Alejandro Fokina', 'Norbert Gombos', 'Marc Polmans', 'Mikhail Kukushkin', 'Kyle Edmund', 'Dusan Lajovic', 'Daniel Evans', 'Mackenzie Mcdonald', 'Yoshihito Nishioka', 'Laslo Djere', 'Tatsuma Ito', 'Prajnesh Gunneswaran', 'Jan Lennard Struff', 'Novak Djokovic' ]

    #Example predictions for the 2020 Australian Open
    preds_df = tML.get_final_features(players, pd.to_datetime('2020-01-20'), deuce_atp_long, last_cols, rolling_cols, 60, 365)
    ML_cols = ML_cols_subset.copy()
    ML_cols.remove('player_1_win')
    preds_df['player_1_win_probability'] = model.predict_proba(preds_df[ML_cols])[:,1]

    #Average predicted win rating for each player
    print(preds_df.groupby('player_1')['player_1_win_probability'].agg('mean').sort_values(ascending=False).head(100))

if __name__ == "__main__":
    # execute only if run as a script
    demo()
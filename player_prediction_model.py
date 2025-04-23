import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playercareerstats, playergamelog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PlayerPredictionModel:
    def __init__(self):
        self.model = None
        self.recent_model = None
        self.scaler = StandardScaler()
        
        # Define feature sets for each target variable
        self.feature_sets = {
            'PTS': [
                'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'FT_PCT', 'USG_PCT', 'OFF_RATING', 'PIE', 
                'EFF', 'PTS_ROLLING_5', 'FGM_ROLLING_5', 'FGA_ROLLING_5',
                'FTM_ROLLING_5', 'FTA_ROLLING_5', 'FG_PCT_ROLLING_5',
                'FT_PCT_ROLLING_5'
            ],
            'AST': [
                'MIN', 'PTS', 'USG_PCT', 'AST_PCT', 'OFF_RATING', 'PIE',
                'EFF', 'AST_ROLLING_5', 'TOV_ROLLING_5'
            ],
            'STL': [
                'MIN', 'STL_PCT', 'DEF_RATING', 'EFF', 'STL_ROLLING_5',
                'PACE', 'OPP_PTS_OFF_TOV'
            ],
            'BLK': [
                'MIN', 'BLK_PCT', 'DEF_RATING', 'EFF', 'BLK_ROLLING_5',
                'OPP_PTS_PAINT'
            ],
            'TOV': [
                'MIN', 'PTS', 'AST', 'USG_PCT', 'TOV_PCT', 'OFF_RATING',
                'TOV_ROLLING_5', 'PACE'
            ]
        }
        
        self.target_columns = ['PTS', 'AST', 'STL', 'BLK', 'TOV']
    
    def fetch_player_data(self, player_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch player career statistics and recent games from NBA API"""
        try:
            # Fetch career stats
            career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
            career_df = career_stats.get_data_frames()[0]
            
            # Get the last 3 seasons instead of just the last one
            career_df = career_df.tail(3)
            
            # Fetch recent game log
            game_log = playergamelog.PlayerGameLog(player_id=player_id)
            recent_df = game_log.get_data_frames()[0]
            
            return career_df, recent_df
        except Exception as e:
            print(f"Error fetching player data: {e}")
            return None, None
    
    def preprocess_data(self, df: pd.DataFrame, is_recent: bool = False) -> pd.DataFrame:
        """Preprocess the player statistics data with improved feature engineering"""
        # Calculate per-game statistics
        stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
        
        if is_recent:
            # For recent games, use the raw values
            for col in stats_columns:
                if col in df.columns:
                    df[f'{col}_PG'] = df[col]
        else:
            # For season stats, calculate per-game stats
            for col in stats_columns:
                if col in df.columns:
                    df[f'{col}_PG'] = df[col] / df['GP']
        
        # Create advanced metrics
        df['EFF'] = (df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - 
                    (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM']) - df['TOV'])
        
        if not is_recent:
            df['EFF'] = df['EFF'] / df['GP']
        
        # Create usage and efficiency metrics
        if is_recent:
            # For recent games, calculate usage based on minutes played
            df['USG_PCT'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) * (df['MIN'] / 5) / df['MIN']
        else:
            # For season stats, use games played
            df['USG_PCT'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) * (df['MIN'] / 5) / df['GP']
        
        # Create other advanced metrics
        df['AST_PCT'] = df['AST'] / (df['FGM'] + 0.5 * df['FG3M'])
        df['STL_PCT'] = df['STL'] / df['MIN'] * 48
        df['BLK_PCT'] = df['BLK'] / df['MIN'] * 48
        df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
        
        # Create offensive and defensive ratings
        df['OFF_RATING'] = df['PTS'] / df['MIN'] * 48
        df['DEF_RATING'] = df['PTS'] / df['MIN'] * 48
        
        # Create player impact estimate
        df['PIE'] = df['PTS'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
        
        # Create rolling averages for field goals and free throws
        for col in ['FGM', 'FGA', 'FTM', 'FTA', 'FG_PCT', 'FT_PCT']:
            if col in df.columns:
                df[f'{col}_ROLLING_5'] = df[col].rolling(window=5, min_periods=1).mean()
        
        # Create rolling averages for other stats
        for col in stats_columns:
            if col in df.columns:
                df[f'{col}_ROLLING_3'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_ROLLING_5'] = df[col].rolling(window=5, min_periods=1).mean()
                df[f'{col}_ROLLING_10'] = df[col].rolling(window=10, min_periods=1).mean()
        
        # Create momentum features
        for col in stats_columns:
            if col in df.columns:
                df[f'{col}_MOMENTUM'] = df[f'{col}_ROLLING_5'] - df[f'{col}_ROLLING_10']
        
        # Create season progression features
        if not is_recent:
            df['SEASON_PROGRESS'] = df.index / len(df)
        
        # Handle missing values
        df = df.ffill().bfill().fillna(0)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for model training"""
        # Select features specific to the target variable
        available_features = [col for col in self.feature_sets[target] if col in df.columns]
        X = df[available_features].values
        
        # Target variable
        y = df[target].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, target: str, is_recent: bool = False):
        """Train the prediction model with improved parameters"""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize XGBoost model with optimized parameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            gamma=0.1,
            random_state=42
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the model
        if is_recent:
            self.recent_model = model
        else:
            self.model = model
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"{'Recent' if is_recent else 'Season'} Model Performance for {target}:")
        print(f"Training R² score: {train_score:.4f}")
        print(f"Testing R² score: {test_score:.4f}")
    
    def predict(self, player_id: str) -> Dict[str, any]:
        """Generate predictions for a player"""
        # Fetch and preprocess data
        season_df, recent_df = self.fetch_player_data(player_id)
        if season_df is None or recent_df is None:
            return None
        
        # Process data
        season_processed = self.preprocess_data(season_df)
        recent_processed = self.preprocess_data(recent_df, is_recent=True)
        
        predictions = {}
        
        # Generate predictions for each target variable
        for target in self.target_columns:
            # Prepare features for this target
            X_season, y_season = self.prepare_features(season_processed, target)
            X_recent, y_recent = self.prepare_features(recent_processed, target)
            
            # Train models
            self.train_model(X_season, y_season, target, is_recent=False)
            self.train_model(X_recent, y_recent, target, is_recent=True)
            
            # Generate predictions
            season_pred = self.model.predict(X_season)
            recent_pred = self.recent_model.predict(X_recent)
            
            # Calculate statistics
            season_std = np.std(season_pred)
            recent_std = np.std(recent_pred)
            
            # Convert to per-game values for season stats
            season_pred_per_game = season_pred[-1] / season_df['GP'].iloc[-1]
            season_std_per_game = season_std / season_df['GP'].iloc[-1]
            season_avg_per_game = np.mean(season_processed[target]) / season_df['GP'].iloc[-1]
            
            # Create prediction dictionaries
            predictions[target] = {
                'season_based': {
                    'predicted_value': float(season_pred_per_game),
                    'confidence_interval': {
                        'lower': float(season_pred_per_game - 1.96 * season_std_per_game),
                        'upper': float(season_pred_per_game + 1.96 * season_std_per_game)
                    },
                    'recent_average': float(season_avg_per_game),
                    'trend': 'increasing' if season_pred_per_game > season_avg_per_game else 'decreasing'
                },
                'recent_form': {
                    'predicted_value': float(recent_pred[-1]),
                    'confidence_interval': {
                        'lower': float(recent_pred[-1] - 1.96 * recent_std),
                        'upper': float(recent_pred[-1] + 1.96 * recent_std)
                    },
                    'recent_average': float(np.mean(recent_processed[target])),
                    'trend': 'increasing' if recent_pred[-1] > np.mean(recent_processed[target]) else 'decreasing'
                }
            }
        
        # Get recent game statistics
        recent_stats = recent_processed.head(15)[['GAME_DATE', 'MATCHUP'] + self.target_columns]
        
        return {
            'season_based_prediction': {
                'points': predictions['PTS']['season_based'],
                'assists': predictions['AST']['season_based'],
                'steals': predictions['STL']['season_based'],
                'blocks': predictions['BLK']['season_based'],
                'turnovers': predictions['TOV']['season_based']
            },
            'recent_form_prediction': {
                'points': predictions['PTS']['recent_form'],
                'assists': predictions['AST']['recent_form'],
                'steals': predictions['STL']['recent_form'],
                'blocks': predictions['BLK']['recent_form'],
                'turnovers': predictions['TOV']['recent_form']
            },
            'last_15_games': recent_stats.to_dict('records')
        }

def main():
    # Example usage
    model = PlayerPredictionModel()
    
    # Example player ID (Nikola Jokić)
    player_id = '203999'
    
    # Generate predictions
    predictions = model.predict(player_id)
    print("\nPlayer Predictions:")
    print(predictions)

if __name__ == "__main__":
    main() 
from nba_api.stats.static import players
from player_prediction_model import PlayerPredictionModel
import sys
import traceback
from datetime import datetime

def format_date(date_str):
    """Format the NBA API date string to a readable format"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        return date.strftime('%b %d, %Y')
    except:
        return date_str

def main():
    try:
        # Find Nikola Jokic's player ID
        print("Searching for Nikola Jokic in the NBA database...")
        player_dict = players.find_players_by_full_name('Nikola Jokic')
        if not player_dict:
            print("Error: Could not find Nikola Jokic in the player database")
            return
        
        jokic_id = player_dict[0]['id']
        print(f"Successfully found Nikola Jokic with ID: {jokic_id}")
        
        # Initialize the prediction model
        print("\nInitializing prediction model...")
        model = PlayerPredictionModel()
        
        # Generate predictions
        print("\nGenerating predictions...")
        predictions = model.predict(jokic_id)
        
        if predictions is None:
            print("Error: Failed to generate predictions")
            return
        
        # Print season-based predictions
        print("\n=== Season-Based Prediction (Last Season) ===")
        season_pred = predictions['season_based_prediction']
        for stat in ['points', 'assists', 'steals', 'blocks', 'turnovers']:
            print(f"\n{stat.upper()}:")
            pred = season_pred[stat]
            print(f"Predicted Value: {pred['predicted_value']:.2f}")
            print(f"95% Confidence Interval: [{pred['confidence_interval']['lower']:.2f}, {pred['confidence_interval']['upper']:.2f}]")
            print(f"Last Season Average: {pred['recent_average']:.2f}")
            print(f"Trend: {pred['trend']}")
        
        # Print recent form predictions
        print("\n=== Recent Form Prediction ===")
        recent_pred = predictions['recent_form_prediction']
        for stat in ['points', 'assists', 'steals', 'blocks', 'turnovers']:
            print(f"\n{stat.upper()}:")
            pred = recent_pred[stat]
            print(f"Predicted Value: {pred['predicted_value']:.2f}")
            print(f"95% Confidence Interval: [{pred['confidence_interval']['lower']:.2f}, {pred['confidence_interval']['upper']:.2f}]")
            print(f"Last 15 Games Average: {pred['recent_average']:.2f}")
            print(f"Recent Trend: {pred['trend']}")
        
        # Print last 15 games details
        print("\n=== Last 15 Games Performance ===")
        print("Date         Matchup              PTS  AST  STL  BLK  TOV")
        print("-" * 60)
        for game in predictions['last_15_games']:
            date = format_date(game['GAME_DATE'])
            print(f"{date:<12} {game['MATCHUP']:<20} {game['PTS']:>3}  {game['AST']:>3}  {game['STL']:>3}  {game['BLK']:>3}  {game['TOV']:>3}")
        
    except Exception as e:
        print("\nAn error occurred:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
# NBA Player Prediction Model

A machine learning model that predicts NBA player statistics using historical and recent performance data. This model uses the NBA API to fetch player data and XGBoost for making predictions.

## Features

- Predicts key player statistics: Points (PTS), Assists (AST), Steals (STL), Blocks (BLK), and Turnovers (TOV)
- Uses both season-based and recent form predictions
- Provides confidence intervals for predictions
- Includes trend analysis (increasing/decreasing)
- Shows detailed last 15 games performance

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/NBA-Player-Prediction-Model.git
cd NBA-Player-Prediction-Model
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

- `player_prediction_model.py`: Main model implementation
- `test_prediction.py`: Example script to test the model
- `requirements.txt`: Project dependencies

## Usage

### Finding Player IDs

To find a player's ID, you can use the NBA API's player search functionality. Here's how to find a player's ID:

1. Visit the NBA API website: https://www.nba.com/stats/players
2. Search for the player you're interested in
3. The player's ID will be in the URL or you can use the NBA API's player search endpoint

Alternatively, you can use the NBA API's Python package to find player IDs programmatically:

```python
from nba_api.stats.static import players

# Find player by full name
player_dict = players.find_players_by_full_name('LeBron James')
player_id = player_dict[0]['id']
```

### Running Predictions

1. Basic usage with the test script:

```bash
python test_prediction.py
```

2. Using the model in your own code:

```python
from player_prediction_model import PlayerPredictionModel

# Initialize the model
model = PlayerPredictionModel()

# Get predictions for a player (using player ID)
predictions = model.predict(player_id)

# Access predictions
print(predictions['season_based_prediction'])
print(predictions['recent_form_prediction'])
print(predictions['last_15_games'])
```

### Available Commands

1. **Create virtual environment**:

```bash
python3 -m venv venv
```

2. **Activate virtual environment**:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run test predictions**:

```bash
python test_prediction.py
```

5. **Deactivate virtual environment**:

```bash
deactivate
```

## Model Details

### Features Used

The model uses a comprehensive set of features for each prediction type:

- **Points (PTS)**: Minutes played, field goals, three-pointers, free throws, usage percentage, offensive rating, and rolling averages
- **Assists (AST)**: Minutes, points, usage percentage, assist percentage, and rolling averages
- **Steals (STL)**: Minutes, steal percentage, defensive rating, and rolling averages
- **Blocks (BLK)**: Minutes, block percentage, defensive rating, and rolling averages
- **Turnovers (TOV)**: Minutes, points, assists, usage percentage, and rolling averages

### Prediction Types

1. **Season-Based Predictions**:

   - Uses the last 3 seasons of data
   - Provides per-game predictions
   - Includes confidence intervals
   - Shows trend analysis

2. **Recent Form Predictions**:
   - Uses the last 15 games
   - Provides game-by-game predictions
   - Includes confidence intervals
   - Shows recent trend analysis

## Example Output

The model provides predictions in the following format:

```python
{
    'season_based_prediction': {
        'points': {
            'predicted_value': 25.5,
            'confidence_interval': {'lower': 23.2, 'upper': 27.8},
            'recent_average': 24.3,
            'trend': 'increasing'
        },
        # ... similar structure for other stats
    },
    'recent_form_prediction': {
        'points': {
            'predicted_value': 26.7,
            'confidence_interval': {'lower': 24.1, 'upper': 29.3},
            'recent_average': 25.8,
            'trend': 'increasing'
        },
        # ... similar structure for other stats
    },
    'last_15_games': [
        # List of last 15 games with detailed stats
    ]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NBA API for providing player statistics
- XGBoost for the machine learning framework
- Scikit-learn for data preprocessing and model evaluation

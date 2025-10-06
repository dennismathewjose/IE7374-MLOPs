import pandas as pd
from app.processor import process_data

def test_process_data_structure():
    # Simulated dataset
    data = {
        'Country name': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Ladder score': [7.1, 6.9, 6.8, 6.7, 6.6, 6.5],
        'Generosity': [0.3, 0.25, 0.2, 0.35, 0.1, 0.05]
    }
    df = pd.DataFrame(data)
    filename = process_data(df)
    
    result = pd.read_csv(filename)
    assert len(result) == 10  # 5 happiest + 5 generous
    assert 'Score' in result.columns
    assert 'Category' in result.columns

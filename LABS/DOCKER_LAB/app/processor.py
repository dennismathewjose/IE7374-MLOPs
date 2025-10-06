import pandas as pd
from app import config
from datetime import datetime

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Top 5 happiest
    top_happy = df[['Country name', 'Ladder score']].sort_values(by='Ladder score', ascending=False).head(5)
    top_happy['Category'] = 'Happiest'
    top_happy.rename(columns={'Ladder score': 'Score'}, inplace=True)

    # Top 5 most generous
    top_generous = df[['Country name', 'Generosity']].sort_values(by='Generosity', ascending=False).head(5)
    top_generous['Category'] = 'Most Generous'
    top_generous.rename(columns={'Generosity': 'Score'}, inplace=True)

    # Combine and save
    result_df = pd.concat([top_happy, top_generous])

     #Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"top5_happy_generous_{timestamp}.csv"

    result_df.to_csv(filename, index=False)
    return filename

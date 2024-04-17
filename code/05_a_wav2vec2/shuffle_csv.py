import os
import pandas as pd

RAW_DATA_FILE = os.path.join('common-voice','validated.tsv')
SHUFFLED_DATA_FILE = os.path.join('common-voice','shuffled.tsv')

if __name__ == '__main__':
    df = pd.read_csv(RAW_DATA_FILE, sep='\t', low_memory=False)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv(SHUFFLED_DATA_FILE, sep='\t', index=False)
import os
import pandas as pd

RAW_DATA_FILE = os.path.join('D:', os.sep, 'Datasets', 'common-voice-16-full','validated.tsv')
SHUFFLED_DATA_FILE = os.path.join('D:', os.sep, 'Datasets', 'common-voice-16-full','shuffled.tsv')

if __name__ == '__main__':
    df = pd.read_csv(RAW_DATA_FILE, sep='\t', low_memory=False)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv(SHUFFLED_DATA_FILE, sep='\t', index=False)
import os
import csv
import string
from tqdm import tqdm
from pydub import AudioSegment
from transformers import pipeline

BASE_PATH = os.path.join('D:', os.sep, 'Datasets', 'common-voice-16-full')
RAW_DATA_FILE = os.path.join(BASE_PATH,'shuffled.tsv')
TRAIN_FILE = os.path.join(BASE_PATH, 'train.csv')
TEST_FILE = os.path.join(BASE_PATH, 'test.csv')
TEST_TRAIN_RATIO = 8 # Jeder 8. Datensatz geht in Test

MAX_SAMPLES = 100000

dialect_map = {
  'Amerikanisches Deutsch': 'Amerikanisch',
  'Bayerisch': 'Bayerisch',
  'Britisches Deutsch': 'Britisch',
  'Deutschland Deutsch': 'Deutsch',
  'Deutschland Deutsch,Alemanischer Akzent,Süddeutscher Akzent': 'Süddeutsch',
  'Deutschland Deutsch,Berliner Deutsch': 'Berlinerisch',
  'Deutschland Deutsch,Hochdeutsch': 'Deutsch',
  'Deutschland Deutsch,Ruhrgebiet Deutsch,West Deutsch': 'Ruhrdeutsch',
  'Deutschland Deutsch,Süddeutsch': 'Süddeutsch',
  'Niederbayerisch': 'Bayrisch',
  'Niedersächsisches Deutsch,Deutschland Deutsch': 'Deutsch',
  'Nordrhein-Westfalen,Bundesdeutsch, Hochdeutsch,Deutschland Deutsch': 'Deutsch',
  'Ostbelgien,Belgien,Belgisches Deutsch': 'Belgisch',
  'Schweizerdeutsch': 'Schweizerdeutsch',
  'Süddeutsch': 'Süddeutsch',
  'Österreichisches Deutsch': 'Österreichisch',
  'Russisch Deutsch': 'Russisch',
  'Französisch Deutsch': 'Französisch',
  'nach Norddeutschland ausgewanderter Franke - hart/weiche Konsonanten verschwimmen, weichere Sprachmelodie als typisch norddeutsche': 'Fränkisch',
  'Niederländisch Deutsch': 'Niederländisch',
  'Deutschland Deutsch,Polnisch Deutsch': 'Polnisch',
  'Dänisch Deutsch': 'Dänisch',
  'Polnisch Deutsch': 'Polnisch',
  'Türkisch Deutsch': 'Türkisch',
  'Tschechisch Deutsch': 'Tschechisch',
  'Norddeutsch': 'Deutsch',
  'Italienisch Deutsch': 'Italienisch',
  'Ungarisch Deutsch,Ruhrdeutsch': 'Ungarisch',
  'Griechisch Deutsch': 'Griechisch',
  'Deutschland Deutsch,Hochdeutsch,Ostwestfälisch': 'Deutsch',
  'Deutschland Deutsch,Hessisch': 'Hessisch',
  'Leichter saarländische Einschlag mit Unschärfe bei ch und sch,Deutschland Deutsch': 'Deutsch',
  'Österreichisches Deutsch,Kärnten,Steiermark': 'Österreichisch',
  'Kraichgauer': 'Kurpfälzisch',
  'Deutschland Ruhrgebiet': 'Ruhrdeutsch',
  'Ruhrpott Deutsch': 'Ruhrdeutsch',
  'Deutschland Deutsch,Österreichisches Deutsch': 'Österreichisch',
  'Ungarisch Deutsch': 'Ungarisch',
  'Kanadisches Deutsch': 'Kanadisch',
  'Österreichisches Deutsch,Lower Austria': 'Österreichisch',
  'Hochdeutsch,Deutschland Deutsch': 'Deutsch',
  'Deutschland Deutsch,Norddeutsch': 'Deutsch',
  'Deutschland Deutsch,leicht Berlinerisch': 'Berlinerisch',
  'Belgisches Deutsch': 'Belgisch',
  'Hochdeutsch': 'Deutsch',
  'Alemannische Färbung,Schweizer Standart Deutsch': 'Schweizerdeutsch',
  'Slowakisch Deutsch': 'Slowakisch',
  'Deutschland Deutsch,Akzentfrei': 'Deutsch',
  'Deutsch/Berlinern,Berlinerisch,klar,zart,feminin': 'Berlinerisch',
  'Deutschland Deutsch,Fränkisch': 'Fränkisch',
  'Finnisch Deutsch': 'Finnisch',
  'starker lettischer Akzent': 'Lettisch',
  'liechtensteinisches Deutscher': 'Lichtensteinisch',
  'Luxemburgisches Deutsch': 'Luxemburgisch',
  'Österreichisches Deutsch,Oberösterreichisches Deutsch': 'Österreichisch',
  'Deutschland Deutsch,Standarddeutsch,Ruhrpott': 'Ruhrdeutsch',
  'Brasilianisches Deutsch': 'Brasilianisch',
  'Schwäbisch Deutsch': 'Schwäbisch',
  'Schweizerdeutsch,Zürichdeutsch': 'Schweizerdeutsch',
  'Deutschland Deutsch,sächsisch': 'Sächsisch',
  'Schwäbisch Deutsch,Deutschland Deutsch': 'Schwäbisch',
  'Akzentfrei': 'Deutsch',
  'Deutschland Deutsch,Saarland Deutsch,Plattdeutsch,Saarländisch': 'Saarländisch',
  'Deutschland Deutsch,Schwäbisch': 'Schwäbisch',
  'Slowakisch Deutsch': 'Slowakisch',
  'Deutschland Deutsch,Türkisch Deutsch': 'Türkisch',
  'Deutschland Deutsch,Britisches Deutsch': 'Britisch',
  'Österreichisches Deutsch,Bayern': 'Österreichisch',
  'Slowenisch Deutsch': 'Slowenisch',
  'Deutschland Deutsch,relativ akzentfrei': 'Deutsch',
  'Deutschland Deutsch,Deutschland Deutsch Hochdeutsch': 'Deutsch',
  'Österreichisches Deutsch,Theaterdeutsch,Wienerisch,Burgenländisch (Süden),Niederösterreich (Mödling)': 'Österreichisch',
  'Israeli': 'Israelisch',
  'Badisch,Allemannisch': 'Badisch',
  'Deutschland Deutsch,Schwäbischer Akzent': 'Schwäbisch',
  'Französisch Deutsch,Deutschland Deutsch': 'Französisch',
  'Deutschland Deutsch,Nordrhein Westfalen': 'Rheinländisch',
  'Chinesisch Deutsch': 'Chinesisch',
  'Belgisches Deutsch,Französisch Deutsch': 'Belgisch',
  'Rheinländich': 'Rheinländisch',
  'Deutschland Deutsch,Kraichgauer,Odenwälder': 'Kurpfälzische',
  'Deutschland Deutsch Fränkisch': 'Fränkisch',
  'Ruhrdeutsch,Deutschland Deutsch': 'Ruhrdeutsch',
  'Deutschland Deutsch,Bayerisches Deutsch': 'Bayrisch',
  'Deutschland Deutsch,Nordhessisch Deutsch': 'Hessisch',
  'Deutschland Deutsch,Hochdeutsch,Hamburgisch Deutsch,Niedersächsisch Deutsch': 'Deutsch',
  'Normales etwas überdeutlich gesprochenes Deutsch,Deutschland Deutsch': 'Deutsch',
  'Österreichisches Deutsch,Tirol': 'Österreichisch',
  'Süddeutsch-Schweizerdeutsch': 'Schweizerdeutsch',
  'Mecklenburgisch,Deutschland Deutsch': 'Mecklenburgisch',
  'Deutschland Deutsch,Arabisches Deutsch': 'Arabisch',
  'Deutschland Deutsch,Hochdeutsch/kein wirklicher Akzent': 'Deutsch',
  'Deutschland Deutsch,Bayrisch': 'Bayrisch',
  'spanischer Akzent': 'Spanisch',
  'Schriftsprache deutsch aus der Schweiz': 'Schweizerdeutsch',
  'Bulgarisch Deutsch': 'Bulgarisch',
  'Österreichisches Deutsch,Niederösterreich, Wien': 'Österreichisch',
  'Normales etwas überdeutlich gesprochenes Deutsch': 'Deutsch',
  'Litauisch Deutsch': 'Litauisch',
  'Deutschland Deutsch,Stuttgart': 'Schwäbisch',
  'Schweizerdeutsch,Deutschland Deutsch': 'Schweizerdeutsch',
  'Deutschland Deutsch,Niederrhein': 'Niederrheinisch',
  'Deutschland Deutsch,Leichter saarländische Einschlag mit Unschärfe bei ch und sch': 'Saarländisch',
  '': 'Deutsch'
}  

# Emotion über ein externes Modell erkennen
EMOTION_MODEL_NAME = 'padmalcom/wav2vec2-large-emotion-detection-german'
emotions = {'anger':0, 'boredom':1, 'disgust':2, 'fear':3, 'happiness':4, 'sadness':5, 'neutral':6}
audio_classifier = pipeline(task='audio-classification', model=EMOTION_MODEL_NAME)

def emotion(audio_file):
  preds = audio_classifier(audio_file)
  max_score = 0
  max_label = 6
  max_label_text = 'neutral'
  for p in preds:
    if p['score'] > max_score and p['score'] > 0.25:
      max_score = p['score']
      max_label = emotions[p['label']]
      max_label_text = p['label']
      print('There is an emotional file:', max_label_text)
  return max_label_text

def prepare_data():
  with open(RAW_DATA_FILE, 'r', encoding='utf8') as f:
    row_count = sum(1 for line in f)
    print('There are', row_count, 'rows in the dataset.')
  
  with open(RAW_DATA_FILE, 'r', encoding='utf8') as f:
    tsv = csv.DictReader(f, delimiter='\t')
    
    if not os.path.exists(os.path.join(BASE_PATH, 'wavs')):
      os.mkdir(os.path.join(BASE_PATH, 'wavs'))
      
    i = 0
    faulty_lines = 0
    train_file_header_written = False
    test_file_header_written = False
    test_count = 0
    train_count = 0
    with open(TRAIN_FILE, 'w', newline='', encoding='utf8') as train_f:
      with open(TEST_FILE, 'w', newline='', encoding='utf8') as test_f:
        try:
          for line in tqdm(tsv, total=row_count):
            if i >= MAX_SAMPLES:
              break              
            formatted_sample = {}
            formatted_sample['file'] = line['path']
            formatted_sample['sentence'] = line['sentence'].translate(str.maketrans('', '', string.punctuation))
            formatted_sample['age'] = line['age']
            formatted_sample['gender'] = line['gender']
            if line['accents'].strip() in dialect_map:
                formatted_sample['accent'] = dialect_map[line['accents'].strip()]
            else:
                print('Key not in dict:', line['accents'].strip())
            
            if (formatted_sample['sentence'] == None or formatted_sample['sentence'] == 'nan' or line['path'] == None or line['sentence'] == None or line['age'] == None or line['gender'] == None or line['accents'] == None or line['path'].strip() == '' or line['sentence'].strip() == '' or line['age'].strip() == '' or line['gender'].strip() == '' or line['accents'].strip() == '' or formatted_sample['sentence'].strip() == ''):
              faulty_lines += 1
              continue

            mp3FullPath = os.path.join(BASE_PATH, 'clips', line['path'])
            filename, _ = os.path.splitext(os.path.basename(mp3FullPath))
            sound = AudioSegment.from_mp3(mp3FullPath)
            if sound.duration_seconds > 0:
                sound = sound.set_frame_rate(16000)
                sound = sound.set_channels(1)
                wav_path = os.path.join(BASE_PATH, 'wavs', filename + '.wav')
                sound.export(wav_path, format='wav')
                formatted_sample['file'] = filename + '.wav'
                
                # emotion classification
                formatted_sample['emotion'] = 'neutral' #emotion(wav_path)

                if i % TEST_TRAIN_RATIO == 0:
                  if not test_file_header_written:
                    test_w = csv.DictWriter(test_f, formatted_sample.keys())
                    test_w.writeheader()
                    test_file_header_written = True
                  test_w.writerow(formatted_sample)
                  test_count += 1
                else:
                  if not train_file_header_written:
                    train_w = csv.DictWriter(train_f, formatted_sample.keys())
                    train_w.writeheader()
                    train_file_header_written = True
                  train_w.writerow(formatted_sample)
                  train_count += 1
                i += 1
        except KeyboardInterrupt:
          print('Keyboard interrupt called. Exiting...')
        
        print('Found', i, 'samples.', train_count, 'in train and', test_count, 'in test.', faulty_lines, 'lines were faulty.')
    
if __name__ == '__main__':
  prepare_data()
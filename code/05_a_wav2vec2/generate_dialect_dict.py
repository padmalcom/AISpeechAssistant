from preprocess_mcv import dialect_map
import csv

s = ""
dialects = []
with open('common-voice-16-full\\train.csv', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader) # skip header
    for i, row in enumerate(spamreader):
        #print("row: ", row[4])
        if not row[4] in dialects:
            #print("This:", row[4])
            s = s + "'" + row[4] +"': " + str(i) + ","
            dialects.append(row[4])
with open('common-voice-16-full\\test.csv', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader) # skip header
    for i, row in enumerate(spamreader):
        #print("row: ", row[4])
        if not row[4] in dialects:
            #print("This:", row[4])
            s = s + "'" + row[4] +"': " + str(i) + ","
            dialects.append(row[4])            
    print(s)
        
#if __name__ == '__main__':
#    s = ""
#    for i, v in enumerate(dialect_map.values()):
#        s = s + "'" + v+"': " + str(i) + ","
#    print(s)
        
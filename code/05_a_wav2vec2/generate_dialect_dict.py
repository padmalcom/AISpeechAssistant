import csv

s = ""
dialects = []
with open('common-voice-16-full\\train.csv', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader) # skip header
    for i, row in enumerate(spamreader):
        if not row[4] in dialects:
            s = s + "'" + row[4] +"': " + str(i) + ","
            dialects.append(row[4])
with open('common-voice-16-full\\test.csv', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader) # skip header
    for i, row in enumerate(spamreader):
        if not row[4] in dialects:
            s = s + "'" + row[4] +"': " + str(i) + ","
            dialects.append(row[4])            
print(s)
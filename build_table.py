import json, csv, sys

maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

source_path = "enron_spam_data.csv"

text = []

with open(source_path, "r", encoding="utf-8") as csvfile:
    data = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in data:
        text.append(row[1])
        text.append(row[2])

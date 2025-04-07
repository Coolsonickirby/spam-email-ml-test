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

output_messages_file = "dataset.json"
words_table_file = "words_table.json"


words_arr = []

spam_messages = []
ham_messages = []

with open(source_path, "r", encoding="utf-8") as csvfile:
    data = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in data:
        words = [x.replace("\n", "").strip() for x in row[1].strip().split(" ")]
        words.extend(x.replace("\n", "").strip() for x in row[2].strip().split(" "))
        words = list(filter(lambda y: y != "", words))
        if row[3] == "spam":
            spam_messages.append(row[2])
        elif row[3] == "ham":
            ham_messages.append(row[2])
        words_arr.extend(words)

# Remove all duplicates
words_arr = list(set(words_arr))

# Save table
with open(words_table_file, "w+") as f:
    json.dump(words_arr, f)

# Save dataset
with open(output_messages_file, "w+") as f:
    json.dump({
        "spam": spam_messages,
        "ham": ham_messages,
    }, f, indent=4)
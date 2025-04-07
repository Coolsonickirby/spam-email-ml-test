import json

def tokenize(x):
    words = [y.strip() for y in x.strip().replace("\n", " ").strip().split(" ")]
    words = list(filter(lambda y: y != "", words))
    tokenized = [words_table[z] if z in words_table else 0 for z in words]
    return tokenized

words_table_path = "words_table.json"
words_table_src = []
words_table = {}
with open(words_table_path, "r") as f:
    words_table_src = json.load(f)

for x in range(len(words_table_src)):
    words_table[words_table_src[x]] = x

if __name__ == "__main__":
    source_dataset = "dataset.json"

    tokenized_dataset_path = "tokenized_dataset.json"

    dataset = {}
    with open(source_dataset, "r") as f:
        dataset = json.load(f)


    spam_dataset = []
    ham_dataset = []

    for x in dataset["spam"]:
        res = tokenize(x)
        if len(res) != 0:
            spam_dataset.append(res)

    for x in dataset["ham"]:
        res = tokenize(x)
        if len(res) != 0:
            ham_dataset.append(res)

    # Save dataset
    with open(tokenized_dataset_path, "w+") as f:
        json.dump({
            "spam": spam_dataset,
            "ham": ham_dataset,
        }, f)
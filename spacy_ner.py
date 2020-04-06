import spacy
import csv
from collections import defaultdict

print("Loading chunked_corpus_df.csv file >>>***")

columns = defaultdict(list)  # each value in each column is appended to a list

with open('chunked_corpus_df.csv', encoding='utf8') as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items():  # go over each column name and value
            columns[k].append(v)  # append the value into the appropriate list
            # based on column name k
texts = columns['text']
# print(type(texts))

print("Loading Spacy NER model >>>***")

nlp = spacy.load("en_core_web_lg")

print("Extracting NER >>>***")

f = open('spacy_entities.csv', 'w', encoding='utf8')
with f:
    fnames = ['doc_id', 'text', 'start', 'end', 'value', 'label']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

    for i in range(0, len(texts)):
        if i % 1000 == 0:
            print(i)
        text = texts[i]
        doc_id = columns['doc_id'][i]
        doc = nlp(text)
        for entity in doc.ents:
            # print("Sentence: ",text)
            writer.writerow({'doc_id': doc_id, 'text': text, 'start': entity.start_char, 'end': entity.end_char,
                             'value': entity.text, 'label': entity.label_})

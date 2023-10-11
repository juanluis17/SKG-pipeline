import json
import nltk
import os
import argparse
import pandas as pd

"""
+-------------+--------+------------------------------------------------------+
| Date        | Author | Description                                          |
+=============+========+======================================================+
|  4-Nov-2022 | JLGM   | - Add directories as arguments						  |
+-------------+--------+------------------------------------------------------+
@modified_by: Juan-Luis García-Mendoza
"""


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_path", type=str, default="../../dataset/vijini/", help="")
    parser.add_argument("--data_output_dir", type=str, default="../../outputs/vijini_input/", help="")
    return parser.parse_args()


def main():
    args = get_args()
    data_path = args.data_path
    data_output_dir = args.data_output_dir

    try:
        os.mkdir(data_output_dir)
        print("Directory ", data_output_dir, " Created ")
    except FileExistsError:
        print("Directory ", data_output_dir, " already exists")

    already_parsed = os.listdir(data_output_dir)
    files_to_parse = [filename for filename in os.listdir(data_path) if filename not in already_parsed]
    print('> total files:', len(os.listdir(data_path)))
    print('> already parsed:', len(already_parsed))
    print('> files_to_parse:', len(files_to_parse))

    for file in sorted(files_to_parse):
        if file[-4:] != '.csv':
            continue
        fw = open(data_output_dir + file, 'w+')
        with open(data_path + file, 'r', encoding='utf-8') as f:
            print('> processing:', file)
            df = pd.read_csv(filepath_or_buffer=f)
            for index, row in df.iterrows():
                id = "{}_{}".format(file.split(".csv")[0], index)
                text = row['text']
                sentences = nltk.sent_tokenize(text)

                # if len(sentences) <= 10:  # maximum abstracts with 15 sentences
                sentences_tokenized = []
                new_sentences = []
                for s in sentences:
                    tokens = [t for t in nltk.word_tokenize(s.encode('utf8', 'ignore').decode('ascii', 'ignore')) if
                              t not in ['']]
                    if len(tokens) >= 5:  # len(tokens) <= 250 and len(tokens) >= 5:
                        sentences_tokenized += [tokens]
                        new_sentences.append(s)

                if len(sentences_tokenized) >= 1:
                    data_input_for_dygepp = json.dump({
                        'clusters': [[] for x in range(len(sentences_tokenized))],
                        'sentences': sentences_tokenized,
                        'sentences_': new_sentences,
                        'ner': [[] for x in range(len(sentences_tokenized))],
                        'relations': [[] for x in range(len(sentences_tokenized))],
                        'doc_key': str(id),
                        'dataset': 'scierc'
                    }, fw)
                    fw.write('\n')

        fw.flush()
        fw.close()


if __name__ == "__main__":
    main()

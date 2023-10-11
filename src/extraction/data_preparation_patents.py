import json
import nltk
import os
import argparse

"""
+-------------+--------+------------------------------------------------------+
| Date        | Author | Description                                          |
+=============+========+======================================================+
|  24-May-2023 | JLGM   | - Processing of list of patent files				  |
+-------------+--------+------------------------------------------------------+
created_by: Juan-Luis Garcia-Mendoza

"""

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_path", type=str, default="../../dataset/transfer/", help="")
    parser.add_argument("--data_output_dir", type=str, default="../../outputs/patent_input/", help="")
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
        fw = open(data_output_dir + file.split(".txt")[0]+".json", 'w+')
        with open(data_path + file, 'r', encoding='utf-8') as f:
            print('> processing:', file)
            id = file.split(".txt")[0]
            lines = f.readlines()
            text = "".join(lines)
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

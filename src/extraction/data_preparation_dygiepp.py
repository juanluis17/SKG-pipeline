import json
import nltk
import os
import argparse

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
    parser.add_argument("data_path", type=str, default="../../dataset/computer_science/", help="")
    parser.add_argument("data_output_dir", type=str, default="../../outputs/dygiepp_input/", help="")
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
        if file[-5:] != '.json':
            continue
        fw = open(data_output_dir + file, 'w+')
        with open(data_path + file, 'r', encoding='utf-8') as f:
            print('> processing:', file)
            for paper_row in f:
                paper = json.loads(paper_row)
                if '_source' in paper.keys():
                    source = paper['_source']
                else:
                    source = paper

                if 'id' in source:
                    paper_id = source['id']
                else:
                    paper_id = ''
                    continue

                if 'papertitle' in source:
                    title = source['papertitle']
                elif 'title' in source:
                    title = paper['title']
                else:
                    title = ''
                    continue

                if 'abstract' in source:
                    abstract = source['abstract']
                else:
                    abstract = ''
                    continue

                sentences = nltk.sent_tokenize(abstract)
                if len(sentences) <= 20:  # maximum abstracts with 15 sentences

                    sentences_tokenized = []
                    for s in sentences:
                        tokens = [t for t in nltk.word_tokenize(s.encode('utf8', 'ignore').decode('ascii', 'ignore')) if
                                  t not in ['']]
                        # sentences: maximum 250 tokens, at least 5 tokens, at maximum 5 dots
                        if len(tokens) <= 250 and len(tokens) >= 5:
                            sentences_tokenized += [tokens]
                    sentences_tokenized = [nltk.word_tokenize(
                        title.encode('utf8', 'ignore').decode('ascii', 'ignore'))] + sentences_tokenized
                    sentences_tokenized = [s for s in sentences_tokenized if
                                           len(s) >= 2]  # no empty sentences after ignoring ascii, at least two tokens

                    if len(sentences_tokenized) >= 1:
                        data_input_for_dygepp = json.dump({
                            'clusters': [[] for x in range(len(sentences_tokenized))],
                            'sentences': sentences_tokenized,
                            'ner': [[] for x in range(len(sentences_tokenized))],
                            'relations': [[] for x in range(len(sentences_tokenized))],
                            'doc_key': str(paper_id),
                            'dataset': 'scierc'
                        }, fw)
                        fw.write('\n')

        fw.flush()
        fw.close()


if __name__ == "__main__":
    main()

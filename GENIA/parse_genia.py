"""
Parsing GENIA dataset
"""

import pathlib
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

def parse(path: pathlib, split_size: float):
    """

    :param path: Path to the folder with XLM files
    :param split_size: What should be the size of eval dataset, e.g. 0.3
    :return: path to parsed train set, path to parsed eval set
    """
    output = []
    for file in path.iterdir():
        data = open(str(file))
        contents = data.read()
        soup = BeautifulSoup(contents, 'lxml')
        sentences = soup.find('abstracttext').findChildren(recursive=False)
        for sen in sentences:
            output.append(sen.get_text())

    # Split into test and train file
    train_dataset, eval_dataset = train_test_split(output, test_size=0.25)

    train_dataset = ' '.join(train_dataset)
    eval_dataset = ' '.join(eval_dataset)

    train_dataset_path = pathlib.Path(__file__).absolute().parent.joinpath('genia_parsed_corpus_train.txt')
    eval_dataset_path = pathlib.Path(__file__).absolute().parent.joinpath('genia_parsed_corpus_eval.txt')

    with open(train_dataset_path, "a", encoding="utf-8") as f:
        f.write(train_dataset)

    with open(eval_dataset_path, "a", encoding="utf-8") as f:
        f.write(eval_dataset)

    return str(train_dataset_path), str(eval_dataset_path)

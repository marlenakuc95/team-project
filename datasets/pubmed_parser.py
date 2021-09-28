import requests
import pathlib
import gzip
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
from argparse import ArgumentParser

ROOT = pathlib.Path(__file__).absolute().parent
ENCODING = "utf-8"
DATAFRAME = ROOT.joinpath('pubmed_pointers.csv')

# Prepare folder for pubmed data storage
pubmed = ROOT.joinpath("pubmed")
pubmed.mkdir(parents=True, exist_ok=True)

pubmed_source = pubmed.joinpath("source")
pubmed_source.mkdir(parents=True, exist_ok=True)

pubmed_ann_parsed = pubmed.joinpath("parsed_annotator")
pubmed_ann_parsed.mkdir(parents=True, exist_ok=True)

pubmed_tr_parsed = pubmed.joinpath("parsed_tr")
pubmed_tr_parsed.mkdir(parents=True, exist_ok=True)


# Process single file
def parse(file_url: str, file_name: str):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f"  File requested:  {file_name}")
    req = requests.request('GET', str(file_url))
    output_path = pathlib.Path(pubmed_source).joinpath(pathlib.Path(file_url).stem)

    # Save a file
    with open(str(output_path), 'wb') as f:
        f.write(req.content)

    # Unzip XML (.gz format)
    with gzip.open(str(output_path), 'rb') as f_in:
        with open(str(output_path)[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f"  Parsing starts:  {file_name}")

    # Parse Abstract
    tree = ET.parse(str(output_path)[:-3])
    root = tree.getroot()

    output_ann = ''
    for PubmedArticle in root:
        try:
            abstract = PubmedArticle.find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text

            pubmed_id = PubmedArticle.find('PubmedData').find('ArticleIdList').find("ArticleId[@IdType='pubmed']").text
            title = PubmedArticle.find('MedlineCitation').find('Article').find('ArticleTitle').text

            output_ann = 'UI  - ' + pubmed_id + '\n' + 'TI  - ' + title + '\n' + 'ABS  - ' + abstract + '\n\n' + output_ann

            # Non-ASCII
            encoded_string = abstract.encode("ascii", "ignore")
            decode_string = encoded_string.decode()

            # Create dir path for tr output
            pubmed_tr_dir = pubmed_tr_parsed.joinpath(file_name)
            pubmed_tr_dir.mkdir(parents=True, exist_ok=True)

            # Save as text file (transformer input)
            output_file = pathlib.Path(pubmed_tr_parsed, pubmed_tr_dir).joinpath(
                pubmed_id + '_parsed').with_suffix(".txt")
            with open(str(output_file), 'w', encoding=ENCODING) as f:
                f.write(decode_string)

        except AttributeError:
            pass

    # Non-ASCII
    encoded_string = output_ann.encode("ascii", "ignore")
    decode_string = encoded_string.decode()

    # Save as text file (annotator)
    output_file = pathlib.Path(pubmed_ann_parsed).joinpath(
        file_name + '_parsed').with_suffix(".txt")
    with open(str(output_file), 'w', encoding=ENCODING) as f:
        f.write(decode_string)


# Process single file
def main(start_idx: int, end_idx: int):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f"  Parsing {start_idx} : {end_idx}")

    # Read from dataframe
    dt = pd.read_csv(DATAFRAME)

    # Slice to indicate which files are to be parsed
    dt_parsed = dt[start_idx:end_idx]

    [parse(x, y) for x, y in zip(dt_parsed['file_url'], dt_parsed['file_name'])]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("start_idx", type=int)
    parser.add_argument("end_idx", type=int)
    args = parser.parse_args()
    start_idx = args.start_idx
    end_idx = args.end_idx

    main(start_idx=start_idx, end_idx=end_idx)


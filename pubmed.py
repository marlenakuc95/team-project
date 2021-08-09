import requests
from bs4 import BeautifulSoup
import pathlib
import gzip
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime

ROOT = pathlib.Path(__file__).absolute().parent
ENCODING = "utf-8"

# Prepare folder for pubmed data storge
pubmed_path = ROOT.joinpath("pubmed_dataset")
pubmed_path.mkdir(parents=True, exist_ok=True)

pubmed_parsed_path = pubmed_path.joinpath("pubmed_parsed")
pubmed_parsed_path.mkdir(parents=True, exist_ok=True)

# Get the page content
URL = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
page = requests.get(URL)

# Get list of links to be downloaded
soup = BeautifulSoup(page.content, 'html.parser')
file_tags = soup.findAll('a')[2:]

urls = []
for url in file_tags:
    urls.append(url['href'])

# Filter out .md5 files
urls = [x for x in urls if not x.endswith('.md5')]


# Process single file
def parse_file(file_url):
    print(file_url)
    req = requests.get(URL + file_url)
    output_path = pathlib.Path(pubmed_path).joinpath(file_url)

    # Save a file
    with open(str(output_path), 'wb') as f:
        f.write(req.content)

    # Unzip XML (.gz format)
    with gzip.open(str(output_path), 'rb') as f_in:
        with open(str(output_path)[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Parse Abstract
    tree = ET.parse(str(output_path)[:-3])
    root = tree.getroot()
    output_schema = ''
    for PubmedArticle in root:
        try:
            abstract = PubmedArticle.find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text
            pubmed_id = PubmedArticle.find('PubmedData').find('ArticleIdList').find("ArticleId[@IdType='pubmed']").text
            title = PubmedArticle.find('MedlineCitation').find('Article').find('ArticleTitle').text

            output_schema = 'UI  - ' + pubmed_id + '\n' + 'TI  - ' + title + '\n' + 'ABS  - ' + abstract + '\n\n' + output_schema

        except AttributeError:
            pass

    # Save as text file
    output_file = pathlib.Path(pubmed_path).joinpath("pubmed_parsed").joinpath(
            pathlib.Path(file_url).stem + '_parsed').with_suffix(".txt")
    with open(str(output_file), 'w', encoding=ENCODING) as f:
        f.write(output_schema)

    # Delete XML file
    file_to_rem = pathlib.Path(str(output_path)[:-3])
    file_to_rem.unlink()


for i in range(len(urls)):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f"  Parsing {i}/{len(urls)}")
    parse_file(urls[i])

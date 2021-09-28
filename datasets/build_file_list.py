"""
For parallel parsing, we build a list of files to be processed. Later, one function can be called on many items from the list at the same time.
"""
import requests
from bs4 import BeautifulSoup
import pathlib
import pandas as pd

URL = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'

def build_pubmed_pointers():

    # Get the page's content
    page = requests.get(URL)

    # Get list of links to be downloaded
    soup = BeautifulSoup(page.content, 'html.parser')
    file_tags = soup.findAll('a')[2:]

    urls = []
    for url in file_tags:
        urls.append(url['href'])

    # Filter out .md5 files and add leading URL
    urls = [pathlib.Path(''.join([URL, x])) for x in urls if not x.endswith('.md5')]

    # Get only the filenames
    filenames = []
    for url in urls:
        filename = pathlib.Path(url)
        while filename.suffix:
            filename = filename.with_suffix('')
        filenames.append(filename.stem)

    # Create a a dataframe with file_url and filename and save
    dt = pd.DataFrame({'file_url': urls, 'file_name': filenames})
    dt.to_csv(pathlib.Path(__file__).absolute().parent.joinpath('pubmed_pointers.csv'))

def main():
    build_pubmed_pointers()
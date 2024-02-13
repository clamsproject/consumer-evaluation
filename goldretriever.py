import json
from pathlib import Path
from urllib.parse import urljoin

import requests


def download_golds(gold_dir_url, folder_name=None):
    import tempfile
    # code adapt from Angela Lam's

    if folder_name is None:
        folder_name = tempfile.TemporaryDirectory().name
    # Create a new directory to store the downloaded files on local computer
    target_dir = Path(folder_name)
    if not target_dir.exists():
        target_dir.mkdir()

    # Check if the directory is empty
    try:
        next(target_dir.glob('*'))
        raise Exception("The folder '" + folder_name + "' already exists and is not empty")
    except StopIteration:
        pass

    # Send a GET request to the repository URL and extract the HTML content
    response = requests.get(gold_dir_url)

    # github responses with JSON? wow
    payload = json.loads(response.text)['payload']
    links = [i['path'] for i in payload['tree']['items']]

    # Download each file in the links list into the created folder
    for link in links:
        raw_url = urljoin('https://raw.githubusercontent.com/',
                          '/'.join((payload['repo']['ownerLogin'],
                                    payload['repo']['name'],
                                    payload['refInfo']['name'],
                                    link)))
        file_path = target_dir / link.split('/')[-1]
        with open(file_path, 'wb') as file:
            response = requests.get(raw_url)
            file.write(response.content)
    return folder_name

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download gold files from a github repository')
    parser.add_argument('-d', '--download_dir', default=None, 
                        help='The name of the folder to store the downloaded files. '
                             'If not provided, a system temporary directory will be created')
    parser.add_argument('gold_url', help='The URL of the gold directory')
    args = parser.parse_args()
    download_golds(args.gold_url, args.download_dir)
    
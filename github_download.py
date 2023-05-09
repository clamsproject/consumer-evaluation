# this code is used to enable download diretly from a github repo for test files and gold standards
# mostly meant for file types like .ann and .mmif files
# author @Angelalam

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# The repository URL, change this 
# Gold standard files: https://github.com/clamsproject/clams-aapb-annotations/tree/main/golds/ner/2022-jun-namedentity
url = 'https://github.com/JinnyViboonlarp/ner-evaluation/tree/main/gold-file-examples'

# Extract the repository name from the URL, name would be the phrase after the last "/"
repo_name = urlparse(url).path.split('/')[-1]

# Create a new directory to store the downloaded files on local computer 
if not os.path.exists(repo_name):
    os.mkdir(repo_name)

# Send a GET request to the repository URL and extract the HTML content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links to .mmif, .txt, .md and .ann files in the HTML content
links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith(('.mmif', '.txt', '.md', '.ann'))]

# Download each file in the links list into the created folder
for link in links:
    raw_url = urljoin('https://raw.githubusercontent.com/', link.replace('/blob/', '/'))
    file_name = os.path.basename(link)
    file_path = os.path.join(repo_name, file_name)
    with open(file_path, 'wb') as file:
        response = requests.get(raw_url)
        file.write(response.content)
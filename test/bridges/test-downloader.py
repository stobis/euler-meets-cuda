import argparse
import requests
import shutil
import pathlib
import subprocess
import os
import stat

# Script command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("address", help="URL to test to download", nargs='?')
args = parser.parse_args()

# Pre-defined sources
sources = {
    'Network Repository': {
        'urlfile': 'networkrepository-urls.txt',
        'folder': 'networkrepository/'
    }
}

# Proper part
def download_one(url, folder):
    url = url.rstrip()
    print('Downloading from... {0} '.format(url), flush=True)
    
    local_filename = url[url.rfind('/')+1:] # TODO
    local_filepath = folder + local_filename

    # with requests.post(url, stream=True, allow_redirects=True) as r:
    #     with open(local_filepath, 'wb') as f:
    #         shutil.copyfileobj(r.raw, f)
    subprocess.run(["wget", "-P" + folder, url])

    subprocess.run(["unzip", "-d" + folder, local_filepath, "*.mtx"])
    subprocess.run(["rm", local_filepath])
    
    os.chmod(os.getcwd() + "/" + local_filepath[:-4] + ".mtx", stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)

    return

def download(source_name):
    print('---- {0} ----'.format(source_name), flush=True)
    # Extract source info
    (urlfile, folder) = (sources[source_name]['urlfile'], sources[source_name]['folder'])
    
    # mkdir for tests
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 

    # dl tests
    with open(urlfile) as file:
        for line in file:
            if not line.startswith('#'):
                download_one(line, folder)
    return

if __name__ == "__main__":
    if args.address is not None:
        # URL is specified
        download_one(args.address, '')
    else:
        # Use pre-defined sources
        for source in sources:
            download(source)

import argparse
import requests
import shutil
import pathlib
import subprocess
import os
import stat
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("type", help="test format type (one of dimacs/mm/snap)", nargs='?')
parser.add_argument("url", help="URL to test to download", nargs='?')
args = parser.parse_args()

config = None

def download_one(format, url):
    url = url.rstrip()
    filename = url[url.rfind('/')+1:] # TODO
    
    print('=== Prepare {1} from... {0} ==='.format(url, filename), flush=True)
    
    if filename + '.bin' in os.listdir('in/') :
        print("{0} already exists. Skipping...".format(filename + 'bin'))
        return

    print('1. Download')
    subprocess.run(["wget", "-nc", "--show-progress", "-q", "-Pin", url])

    print('2. Extract')
    if filename.endswith('.tar.gz'):
        subprocess.run(["tar", "--strip-components=1", "-xzf", "in/" + filename, "-C", "in/", "--wildcards", "*.mtx"])
        subprocess.run(["rm", "in/" + filename])
    else:
        subprocess.run(["gzip", "-d", "in/" + filename])    
    
    print('3. Convert to proper input format')
    # os.chmod(os.getcwd() + "/" + local_filepath[:-4] + ".mtx", stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)
    return

def download_all():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        for format in cfg:
            for url in cfg[format]:
                download_one(format, url)

def read_config():
    global config
    with open("config.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)

if __name__ == "__main__":
    read_config()
    print(config)
    if args.type != None and args.url != None:
        download_one(args.type, args.url)
    else:
        download_all()

#!/usr/bin/python3

import argparse
import requests
import shutil
import pathlib
import subprocess
import os
import stat
import yaml
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("type", help="test format type (one of dimacs/mm/snap)", nargs='?')
parser.add_argument("url", help="URL to test to download", nargs='?')
args = parser.parse_args()

config = None

def download_one(format, url):
    url = url.rstrip()
    filename = os.path.basename(url) # url[url.rfind('/')+1:] # TODO
    testfilename = re.sub('.gz', '', filename)
    testfilename = re.sub('.tar', '.mtx', testfilename)

    print('=== Source: {0} ==='.format(url), flush=True)
    # print(filename,testfilename)

    if testfilename + '.bin' in os.listdir('in/') :
        print("{0} already exists. Skipping...\n".format(filename + '.bin'))
        return

    print('1. Download')
    subprocess.run(["wget", "-nc", "--show-progress", "-q", "-Pin", url])

    print('2. Extract')
    if filename.endswith('.tar.gz'):
        subprocess.run(["tar", "--strip-components=1", "-xzf", "in/" + filename, "-C", "in/", "--wildcards", "*.mtx"])
    else:
        subprocess.run(["gzip", "-d", "in/" + filename])
    subprocess.run(["rm", "-f", "in/" + filename])
    
    print('3. Convert to proper input format')
    os.chmod(os.getcwd() + "/in/" + testfilename, stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)
    subprocess.run(["./{}2ecl.e".format(format), "in/{0}".format(testfilename), "in/{0}.bin.tmp".format(testfilename)])
    
    subprocess.run(["rm", "-f", "in/" + testfilename])

    print('4. Convert input graph to connected')
    subprocess.run(["./connect.e", "in/{0}.bin.tmp".format(testfilename), "in/{0}.bin".format(testfilename)])
    subprocess.run(["rm", "-f", "in/{0}.bin.tmp".format(testfilename)])

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
    # print(config)
    Path("./in").mkdir(exist_ok=True)
    if args.type != None and args.url != None:
        download_one(args.type, args.url)
    else:
        download_all()

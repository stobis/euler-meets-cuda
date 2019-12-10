import argparse
import requests
import shutil
import pathlib
import os
import subprocess

# Script command-line arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Pre-defined sources
sources = {
    'Network Repository': {
        'parser': '../networkrepository-parser.e',
        'folder': 'networkrepository/',
        'ext': 'mtx'
    }
}

# Misc
path_preffix = ''
parsed_ext = '.bin'

# Proper part
def parse_one(parser, folder, file):
    print('Parsing... {0} '.format(file), end='')
    
    parsed_filename = file + parsed_ext
    
    file_path = path_preffix + folder + file
    parsed_path = path_preffix + folder + parsed_filename
    subprocess.run([parser, file_path, parsed_path])

    print('to file ({0}) DONE.'.format(parsed_filename))
    return

def parse(source_name):
    print('---- {0} ----'.format(source_name))
    # Extract source info
    (parser, folder, ext) = (
        sources[source_name]['parser'], 
        sources[source_name]['folder'],
        sources[source_name]['ext']
    )
    
    # mkdir for tests
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 

    # parse tests
    for file in os.listdir(folder):
        if file.endswith(ext):
            parse_one(parser, folder, file)
    return

if __name__ == "__main__":
    for source in sources:
        parse(source)

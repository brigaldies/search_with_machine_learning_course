import sys
import os
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

# Location for category data
categoriesFilename = '/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
categoriesOutput = '/workspace/datasets/product_data/categories/categories.csv'
parser = argparse.ArgumentParser(description='Category viewer')
general = parser.add_argument_group("general")
general.add_argument("--max_depth", default=0, type=int, help="optional arg to specify max depth of category tree")
args = parser.parse_args()
maxDepth = args.max_depth
tree = ET.parse(categoriesFilename)
root = tree.getroot()

catPathStrs = set()

with open(categoriesOutput, 'w') as f:
    f.write(f"name\tid\tpath\n")    
    for child in root:
        catName = child.find('name').text
        catId = child.find('id').text
        catPath = child.find('path')
        catPathStr = ''
        depth = 0
        for cat in catPath:
            if catPathStr != '':
                catPathStr = catPathStr + ' > '
            catPathStr = catPathStr + f"{cat.find('name').text} [{cat.find('id').text}]"
            depth = depth + 1
            catPathStrs.add(catPathStr)
            if maxDepth > 0 and depth == maxDepth:
                break
        f.write(f"{catName}\t{catId}\t{catPathStr}\n")        

# Sort for readability
for catPathStr in sorted(catPathStrs):
    print(catPathStr)

print(f"Categories written to {categoriesOutput}")
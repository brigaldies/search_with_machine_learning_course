import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import nltk
from nltk.stem import SnowballStemmer
nltk.download("punkt")
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")

DEBUG = False

def transform_name(product_name):
    clean_product_name = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(product_name)])
    if DEBUG: print(f"transform: {product_name} --> {clean_product_name}")
    return clean_product_name

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENTED: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

general.add_argument("--categories_depth", default=-1, type=int, help="The depth at which to extract the category labels. -1 means full depth.")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

categories_file = output_dir.joinpath("categories_names.txt")

if args.input:
    directory = args.input
# IMPLEMENTED:  Track the number of items in each category and only output if above the min
categories_dict = {}
min_products = args.min_products
sample_rate = args.sample_rate
categories_depth = args.categories_depth
if categories_depth == 0:
    print(f"WARNING: categories_depth={categories_depth} must be -1 (full depth) or a positive number. Using the full depth mode.")
    categories_depth = -1

print(f"min_products={min_products}, sample_rate={sample_rate}, categories_depth={categories_depth}")
print(f"Writing labeled categories to: {output_file}")
print(f"Writing categories names to  : {categories_file}")

for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        print(f"Processing {filename}")
        f = os.path.join(directory, filename)
        tree = ET.parse(f)
        root = tree.getroot()
        products_count = 0
        for child in root:
            if random.random() > sample_rate:
                continue
            # Check to make sure category name is valid
            if (child.find('name') is not None and child.find('name').text is not None and
                child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):

                products_count += 1

                # Replace newline chars with spaces so fastText doesn't complain
                product_name = child.find('name').text.replace('\n', ' ')

                # Determine the hierarchy path extraction depth based on command-line arg "--categories_depth"
                category_path_depth = len(child.find('categoryPath'))
                extract_category_depth = category_path_depth # Max depth by default
                if categories_depth > 0:
                    extract_category_depth = min(category_path_depth, categories_depth)

                if DEBUG: print(f"extract_category_depth={extract_category_depth}")

                category_leaf = child.find('categoryPath')[extract_category_depth - 1]
                cat_id = category_leaf[0].text
                cat_name = category_leaf[1].text

                if cat_id not in categories_dict.keys():
                    categories_dict[cat_id] = {
                        'name': cat_name,
                        'count': 1,
                        'products': []
                    }
                else:
                    categories_dict[cat_id]['count'] += 1

                categories_dict[cat_id]['products'].append(product_name)

        print(f"\t{products_count} products processed.")

excluded_count = 0
with open(output_file, 'w') as output:
    with open(categories_file, 'w') as categories_output:
        for k,v in categories_dict.items():
            excluded = True
            if v['count'] >= min_products:
                excluded = False
                products = v['products']
                for product in products:
                    product_name_transformed = transform_name(product)
                    label_to_product_name = f"__label__{k} {product_name_transformed}"
                    output.write(f"{label_to_product_name}\n")
            else:
                excluded_count += 1

            categories_output.write(f"__label__{k}\t{v['count']}\t{excluded}\t{v['name']}\n")

print(f"{excluded_count} categories excluded")
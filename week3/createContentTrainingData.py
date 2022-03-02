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

def transform_name(product_name):
    clean_product_name = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(product_name)])
    print(f"transform: {product_name} --> {clean_product_name}")
    return clean_product_name

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

categories_file = output_dir.joinpath("categories_names.txt")

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate

print("Writing labeled categories to %s" % output_file)
print("Writing categories to %s" % categories_file)
with open(output_file, 'w') as output:
    with open(categories_file, 'w') as categories_output:
        for filename in os.listdir(directory):
            if filename.endswith(".xml"):
                print("Processing %s" % filename)
                f = os.path.join(directory, filename)
                tree = ET.parse(f)
                root = tree.getroot()
                for child in root:
                    if random.random() > sample_rate:
                        continue
                    # Check to make sure category name is valid
                    if (child.find('name') is not None and child.find('name').text is not None and
                        child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                        child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                        # Choose last element in categoryPath as the leaf categoryId
                        category_leaf = child.find('categoryPath')[len(child.find('categoryPath')) - 1]
                        cat_id = category_leaf[0].text
                        cat_name = category_leaf[1].text
                        # Replace newline chars with spaces so fastText doesn't complain
                        name = child.find('name').text.replace('\n', ' ')
                        output.write("__label__%s %s\n" % (cat_id, transform_name(name)))
                        categories_output.write("%s: %s\n" % (cat_id, cat_name))


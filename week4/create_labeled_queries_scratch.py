import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import time
import datetime

# Useful if you want to perform stemming.
import nltk
nltk.download("punkt")
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = nltk.stem.PorterStemmer()

DEBUG = False

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]

# Keep queries with leaf categories only
df = df[df['category'].isin(categories)]

# Grab the categories' parents via a merge
# df = df.merge(parents_df, 'left', 'category', indicator = True)

# Data QC: Assert that every category has a parent
#assert df[df['_merge'] != 'both'].shape[0] == 0
#df.drop(columns=['_merge'], inplace = True)

# Initialize the LABEL to the category as read from train.csv
df['label'] = df['category']

# Rollup audit column
df['rollups'] = df['label']

# Get label's parent
df = df.merge(parents_df, 'left', left_on = 'label', right_on = 'category', suffixes = (None, "_merged_right"), indicator = True)
assert df[(df['_merge'] != 'both') & (df['label'] != root_category_id)].shape[0] == 0
df.drop(columns=['_merge', 'category_merged_right'], inplace = True) 

# IMPLEMENTED: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalize_query(query_text):
    normalized_query = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(query_text)])
    if DEBUG: print(f"Query normalization: {query_text} --> {normalized_query}")
    return normalized_query

# df['normalized_query'] = df.apply(lambda row: normalize_query(row['query']), axis = 1)

# [BEGIN] IMPLEMENTING: Roll up categories to ancestors to satisfy the minimum number of queries per category.
MAX_LOOP_COUNT = 10
QUERY_COUNT_THRESHOLD = 100

# First, "walk" the loop to see what needs to be done

# --------------------------
# 1st pass
# Group by label
grouped_by_label_df = df[['label', 'query']].groupby(['label'], as_index = False).count()
grouped_by_label_df.columns = ['label', 'query_count']
df = df.merge(grouped_by_label_df, 'left', 'label', indicator = True)

# Data QC: Assert that every category has a query count
assert df[df['_merge'] != 'both'].shape[0] == 0
df.drop(columns=['_merge'], inplace = True)

# First pass
labels_under_threshold_df = grouped_by_label_df[grouped_by_label_df['query_count'] < QUERY_COUNT_THRESHOLD]
df = df.merge(labels_under_threshold_df, 'left', 'label', suffixes = (None, "_under_threshold"), indicator = True)

# Roll up!
# Set the label to the parent when the category's query count < THRESHOLD
df['label'] = df.apply(lambda row: row['parent'] if (row['_merge'] == 'both' and not pd.isnull(row['parent'])) else row['label'], axis = 1)
df['rollups'] = df.apply(lambda row: row['parent'] + " > " + row['rollups'] if (row['_merge'] == 'both' and not pd.isnull(row['parent'])) else row['rollups'], axis = 1)

# --------------------------
# 2nd pass
# Reset
df.drop(columns=['parent', '_merge', 'query_count', 'query_count_under_threshold'], inplace = True)

# Update label's parent
df = df.merge(parents_df, 'left', left_on = 'label', right_on = 'category', suffixes = (None, "_merged_right"), indicator = True)
assert df[(df['_merge'] != 'both') & (df['label'] != root_category_id)].shape[0] == 0
df.drop(columns=['_merge', 'category_merged_right'], inplace = True) 

# Group by label
grouped_by_label_pass_2_df = df[['label', 'query']].groupby(['label'], as_index = False).count()
grouped_by_label_pass_2_df.columns = ['label', 'query_count']
df = df.merge(grouped_by_label_pass_2_df, 'left', 'label', indicator = True)
assert df[df['_merge'] != 'both'].shape[0] == 0
df.drop(columns=['_merge'], inplace = True)

# Find the under-threshold labels
labels_under_threshold_pass_2_df = grouped_by_label_pass_2_df[grouped_by_label_pass_2_df['query_count'] < QUERY_COUNT_THRESHOLD]
df = df.merge(labels_under_threshold_pass_2_df, 'left', 'label', suffixes = (None, "_under_threshold"), indicator = True)

# Roll up
df['label'] = df.apply(lambda row: row['parent'] if (row['_merge'] == 'both' and not pd.isnull(row['parent'])) else row['label'], axis = 1)
df['rollups'] = df.apply(lambda row: row['parent'] + " > " + row['rollups'] if (row['_merge'] == 'both' and not pd.isnull(row['parent'])) else row['rollups'], axis = 1)

# End walking the loop

# [END] IMPLEMENTING: Roll up categories to ancestors to satisfy the minimum number of queries per category.

# Create labels in fastText format.
df['fasttext_label'] = '__label__' + df['label']

# Normalize the queries
print(f"Normalizing {df.shape[0]} queries...")
time_start = time.time()
df['normalized_query'] = df.apply(lambda row: normalize_query(row['query']), axis = 1)
print(f"... in {datetime.timedelta(seconds=time.time() - time_start)}")

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
# df = df[df['category'].isin(categories)]
df['output'] = df['fasttext_label'] + ' ' + df['normalized_query']

print(f"Writing train data to {output_file_name}...")
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
print(f"... in {datetime.timedelta(seconds=time.time() - time_start)}")

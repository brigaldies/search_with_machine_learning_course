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
min_queries = int(args.min_queries)

print(f"min_queries={min_queries}, output_file_name={output_file_name}")

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
time_start = time.time()
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

print(f"Processed {len(categories)} categories from {categories_file_name} in {datetime.timedelta(seconds=time.time() - time_start)}")

parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# IMPLEMENTED: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalize_query(query_text):
    normalized_query = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(query_text)])
    if DEBUG: print(f"Query normalization: {query_text} --> {normalized_query}")
    return normalized_query

# IMPLEMENTED: Roll up categories to ancestors to satisfy the minimum number of queries per category.
MAX_LOOP_COUNT = 10
QUERY_COUNT_THRESHOLD = 100

def prune_categories(queries_file_name, parents_df, max_loop_count = -1, query_count_threshold = QUERY_COUNT_THRESHOLD):
    print(f"Pruning {queries_file_name} with max_loop_count={max_loop_count}, query_count_threshold={query_count_threshold}")
    time_start = time.time()
    # Load the training query/category data
    print(f"Loading the train data from {queries_file_name}...")
    df = pd.read_csv(queries_file_name)[['category', 'query']]

    # Keep queries with leaf categories only
    df = df[df['category'].isin(categories)]

    # Initialize the LABEL to the category as read from train.csv
    df['label'] = df['category']

    # Rollup audit column
    df['audit'] = df['label']

    loop_count = 0
    early_exit = True
    # Use max_loop_count = -1 to prevent early exit
    if max_loop_count < 0:
        print(f"WARN: No safety max loop count!")
    while max_loop_count < 0 or loop_count < max_loop_count:
        loop_count += 1
        print(f"Loop {loop_count}:")

        # Get (first time in the loop)/update (subsequent loops) label's parent
        df = df.merge(parents_df, 'left', left_on = 'label', right_on = 'category', suffixes = (None, "_merged_right"), indicator = True)
        assert df[(df['_merge'] != 'both') & (df['label'] != root_category_id)].shape[0] == 0
        df.drop(columns=['_merge', 'category_merged_right'], inplace = True) 

        # Group by label to count the number of queries per label so far
        grouped_by_label_df = df[['label', 'query']].groupby(['label'], as_index = False).count()
        grouped_by_label_df.columns = ['label', 'query_count']
        df = df.merge(grouped_by_label_df, 'left', 'label', indicator = True)
        assert df[df['_merge'] != 'both'].shape[0] == 0
        df.drop(columns=['_merge'], inplace = True)
        print(f"\t{grouped_by_label_df.shape[0]} unique categories")

        # Identify the labels whose query count < threshold
        labels_under_threshold_df = grouped_by_label_df[grouped_by_label_df['query_count'] < query_count_threshold]

        # Break is all labels' query counts are > threshold
        if labels_under_threshold_df.shape[0] == 0:
            print(f"\tNo label left < {query_count_threshold}")
            print(f"\tEnded with {grouped_by_label_df.shape[0]} categories")
            early_exit = False
            break
        else:
            print(f"\t{labels_under_threshold_df.shape[0]} labels' query counts are < {query_count_threshold}")

        df = df.merge(labels_under_threshold_df, 'left', 'label', suffixes = (None, "_under_threshold"), indicator = True)

        # Roll up: Set the label to the parent when the category's query count < threshold
        print(f"\t\tRolling up...")
        df['label'] = df.apply(lambda row: row['parent'] if (row['_merge'] == 'both' and not pd.isnull(row['parent'])) else row['label'], axis = 1)
        # 'audit' is an audit column to show the successive rollup(s)
        print(f"\t\tAuditing...")
        df['audit'] = df.apply(lambda row: row['parent'] + " > " + row['audit'] if (row['_merge'] == 'both' and not pd.isnull(row['parent'])) else row['audit'], axis = 1)

        # Reset
        df.drop(columns=['parent', '_merge', 'query_count', 'query_count_under_threshold'], inplace = True)

    if early_exit:
        print(f"WARN: Early exist after {loop_count} loops!")
    print(f"Processed {df.shape[0]} queries in {datetime.timedelta(seconds=time.time() - time_start)}")
    return df

pruned_df = prune_categories(queries_file_name, parents_df, query_count_threshold = min_queries)
# [END] IMPLEMENTING: Roll up categories to ancestors to satisfy the minimum number of queries per category.

# Create labels in fastText format.
pruned_df['fasttext_label'] = '__label__' + pruned_df['label']

# Normalize the queries
print(f"Normalizing {pruned_df.shape[0]} queries...")
time_start = time.time()
pruned_df['normalized_query'] = pruned_df.apply(lambda row: normalize_query(row['query']), axis = 1)
print(f"... in {datetime.timedelta(seconds=time.time() - time_start)}")

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
# df = df[df['category'].isin(categories)]
pruned_df['output'] = pruned_df['fasttext_label'] + ' ' + pruned_df['normalized_query']

print(f"Writing train data to {output_file_name}...")
pruned_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
print(f"... in {datetime.timedelta(seconds=time.time() - time_start)}")

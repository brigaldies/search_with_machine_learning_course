import fasttext
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
nltk.download("punkt")
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")

model_file = "/workspace/datasets/fasttext/title_model.bin"
print(f"\n***** Loading model {model_file} ...")
model = fasttext.load_model(model_file)
print(f"Model {model} loaded.")

product_types = [
    "laptops",
    "monitors",
    "keyboards",
    "printers",
    "headphones"
]

brands = [
    "Apple",
    "Toshiba",
    "HP",
    "Sony",
    "Dell"
]

models = [
    "iPhone",
    "ipad",
    "ThinkPad",
    "macbook",
    "inspiron"
]

attributes = [
    "black",
    "portable",
    "battery",
]

k = 10

# TODO: Add argument
stem = False

def find_nn(terms_list):
    for term in terms_list:
        if stem:
            analyzed_term = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(term)])
            print(f"\ttransform: {term} --> {analyzed_term}")
        else:
            analyzed_term = " ".join([token.lower() for token in tokenizer.tokenize(term)])
        
        neighbors = model.get_nearest_neighbors(analyzed_term, k=k)
        print(f"\n{analyzed_term} (raw={term}) neighbors (k={k}):")
        for neighbor in neighbors:
            neighbor_score = neighbor[0]
            neighbor_name = neighbor[1]
            print(f"\t{neighbor_name} ({neighbor_score})")

find_nn(product_types)
find_nn(brands)
find_nn(models)
find_nn(attributes)
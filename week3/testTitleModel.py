import argparse
import fasttext
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
nltk.download("punkt")
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")

model_file = "/workspace/datasets/fasttext/title_model.bin"

parser = argparse.ArgumentParser(description='Test titles model')
general = parser.add_argument_group("general")
general.add_argument("--model", default=model_file, help="Model file")
general.add_argument("--stem", action="store_true", help='Stem input prior to searching for the nearest neighbors')
general.add_argument("--k", default=10, type=int, help='K nearest neighbors')
general.add_argument("--threshold", default=0.9, type=float, help="Nearest neighbor distance threshold")
args = parser.parse_args()
model_file = args.model
stem = args.stem
k = args.k
nn_threshold = args.threshold

print(f"\n***** Loading model {model_file} with...")
model = fasttext.load_model(model_file)
print(f"Model {model} loaded.")
print(f"stem={stem}, k={k}, nn_threashold={nn_threshold}")

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

others = [
    "fabshell",
    "hurley",
    "bluetrek",
    "energi"
]

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
            if neighbor_score >= nn_threshold:
                print(f"\t{neighbor_name} ({neighbor_score})")

print(f"\n***** Product types:")
find_nn(product_types)

print(f"\n***** Brands:")
find_nn(brands)

print(f"\n***** Models:")
find_nn(models)

print(f"\n***** Attributes:")
find_nn(attributes)

print(f"\n***** Others:")
find_nn(others)
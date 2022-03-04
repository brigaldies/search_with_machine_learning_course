import fasttext
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
nltk.download("punkt")
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")

categories_info_file = "/workspace/datasets/categories/categories_names.txt"
print(f"\n***** Loading categories information from {categories_info_file} ...")
categories_df = pd.read_csv(categories_info_file, sep="\t", header=None)
categories_df.columns = ["label", "count", "excluded", "category_name"]

categories_df.info()
print(categories_df.head())
print(f"{categories_df[categories_df['excluded'] == True].shape[0]} categories excluded")

train_file = "/workspace/datasets/categories/categories.fasttext.shuffled"
print(f"\n***** Loading the train data from {train_file} ...")
train_df = pd.read_csv(train_file, header=None)
train_df.columns = [ "labelled_product_name" ]
train_df['label'] = train_df.apply(lambda x: x[0].split()[0], axis = 1)
train_df.info()
print(train_df.head())

model_file = "/workspace/search_with_machine_learning_course/model_categories.bin"
print(f"\n***** Loading model {model_file} ...")
model = fasttext.load_model(model_file)
print(f"Model {model} loaded.")

# TODO: Implement loop to collect product's name from stdin
print(f"\n***** Predicting...")
while True:
    product_name = input("\n---------\nEnter a product name (or \"exit\" to exit): ") 
    if product_name == "exit":
        break

    clean_product_name = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(product_name)])
    print(f"\ttransform: {product_name} --> {clean_product_name}")

    prediction = model.predict(clean_product_name)

    print(f"\tModel's prediction: {prediction}")
    predicted_label = prediction[0][0]
    print(f"\tPredicted label = '{predicted_label}'")

    print(f"\nLookup category for predicted label {predicted_label}:")
    print(categories_df[categories_df['label'] == predicted_label])

    print(f"\nLookup train data for predicated label {predicted_label}:")
    print(train_df[train_df['label'] == predicted_label])

print("\nBye")
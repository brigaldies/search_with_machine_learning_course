# Project 3 Report

## Level 1: Classifying Product Names to Categories

### Prepare Train & Test Data

#### Extraction from the Products XML Files
```
python week3/createContentTrainingData.py --output /workspace/datasets/categories/categories.fasttext --min_products 10
```

```transform_product```:

- Punctuations removal
- Lower casing
- English stemming

```python
import nltk
from nltk.stem import SnowballStemmer
nltk.download("punkt")
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")


def transform_name(product_name):
    clean_product_name = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(product_name)])
    print(f"transform: {product_name} --> {clean_product_name}")
    return clean_product_name
```

#### Shuffle
```shell
shuf /workspace/datasets/categories/categories.fasttext --output /workspace/datasets/categories/categories.fasttext.shuffled
```

#### Train Data
```shell
head -n 50000 /workspace/datasets/categories/categories.fasttext.shuffled > /workspace/datasets/categories/categories.train
```

##### Train Data EDA

```python
import pandas as pd

train_df = pd.read_csv("categories.train", header=None)
train_df.info()
train_df.head()
train_df.columns = ['sample']
train_df['label'] = train_df.apply(lambda x: x['sample'].split()[0], axis = 1)
train_df.info()
train_df.head()
```

#### Test Data
```shell
tail -n 50000 /workspace/datasets/categories/categories.fasttext.shuffled > /workspace/datasets/categories/categories.train
```

### Train Model
```shell
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/categories/categories.train -output model_categories -lr 1.0 -epoch 20 -wordNgrams 2

Read 0M words
Number of words:  15892
Number of labels: 1273
Progress: 100.0% words/sec/thread:     802 lr:  0.000000 avg.loss:  0.493649 ETA:   0h 0m 0s
```

### Test Model
```
~/fastText-0.9.2/fasttext test model_categories.bin /workspace/datasets/categories/categories.test

N       50000
P@1     0.77
R@1     0.77
```

### Test the model with ad hoc examples

```
~/fastText-0.9.2/fasttext predict model_categories.bin -
```

## Level 2: Derive Synonyms from Content

### Run extractTitles.py as is

```shell
python ./week3/extractTitles.py

gitpod /workspace/search_with_machine_learning_course $ head /workspace/datasets/fasttext/titles.txt
NuForce - Icon uDAC-2 USB Audio Receiver and Digital-to-Analog Converter - Red
NuForce - Icon uDAC-2 USB Audio Receiver and Digital-to-Analog Converter - Silver
Sungale - Beam E-Reader - White
Wipeout 2048 - PS Vita
Akai - Refurbished Professional 25-Key Keyboard Controller
Memorex - Slim Jewel Cases (50-pack) - Assorted
Outdoors Unleashed: Africa 3D - Nintendo 3DS
Heavy Fire: Afghanistan - Nintendo Wii
Apple - $25 iTunes Gift Card
Toshiba - Satellite Laptop / Intel® Core™ i3 Processor / 17.3" Display - Matrix Graphite

(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ wc -l /workspace/datasets/fasttext/titles.txt
11620 /workspace/datasets/fasttext/titles.txt
```

### Unsupervised Model Training

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model
Read 0M words
Number of words:  2253
Number of labels: 0
Progress: 100.0% words/sec/thread:    3535 lr:  0.000000 avg.loss:  2.629022 ETA:   0h 0m 0s
```

#### Ad-Hoc Testing

See test_titles_model.sh.log

## Level 3: Integrating Synonyms with Search (Optional but highly encouraged)

## Level 4: ...
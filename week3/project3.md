# Week 3 Project Report

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

I inspected the content of the extracted titles by loading them into a Pandas dataframe, as shown below:

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

One at a time:

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

See also ```test_titles_model.sh.log```.

## Level 3: Integrating Synonyms with Search

### Extract Phone Products Names

```shell
gitpod /workspace/search_with_machine_learning_course $ pyenv activate search_with_ml_week3
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ python ./week3/extractTitles.py --input /workspace/search_with_machine_learning_course/week3/phone_products --sample_rate 1.0  --output /workspace/datasets/fasttext/phone_titles_2.txt
[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
Reading products from /workspace/search_with_machine_learning_course/week3/phone_products
Sample rate=1.0, stem=False
Writing results to /workspace/datasets/fasttext/phone_titles_2.txt
4862 titles extracted.
```

### Train the Phone Products Titles fastText Model

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/phone_titles.txt -output /workspace/datasets/fasttext/phone_model -epoch 25
Read 0M words
Number of words:  691
Number of labels: 0
Progress: 100.0% words/sec/thread:   12917 lr:  0.000000 avg.loss:  2.139039 ETA:   0h 0m 0s
```

### Ad-Hoc Testing of the Phone Products Titles fastText Model

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ./week3/test_titles_model.sh -m /workspace/datasets/fasttext/phone_model.bin -t 0.9
++ python ./week3/testTitleModel.py --model /workspace/datasets/fasttext/phone_model.bin --threshold 0.9
[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Package punkt is already up-to-date!

***** Loading model /workspace/datasets/fasttext/phone_model.bin with...
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Model <fasttext.FastText._FastText object at 0x7f3d9ea85280> loaded.
stem=False, k=10, nn_threashold=0.9

***** Product types:

laptops (raw=laptops) neighbors (k=10):
        laptop (0.9946858286857605)
        6 (0.9141654372215271)

monitors (raw=monitors) neighbors (k=10):

keyboards (raw=keyboards) neighbors (k=10):
        concord (0.9274969100952148)
        keystone (0.9183887243270874)
        eco (0.9176569581031799)
        nauticase (0.9095285534858704)

printers (raw=printers) neighbors (k=10):

headphones (raw=headphones) neighbors (k=10):
        earbud (0.977368175983429)
        earpollution (0.9526700973510742)
        jammin (0.9261226058006287)
        house (0.9165775775909424)
        ozone (0.9115431904792786)
        luxe (0.9076652526855469)
        microphone (0.901847779750824)

***** Brands:

apple (raw=Apple) neighbors (k=10):
        iphone (0.945033073425293)
        kapok (0.9223731160163879)
        4 (0.9205688238143921)
        teal (0.9161614179611206)
        4s (0.9102899432182312)

toshiba (raw=Toshiba) neighbors (k=10):

hp (raw=HP) neighbors (k=10):
        organizers (0.9918403625488281)
        ipaq (0.979442298412323)
        handheld (0.9788981676101685)
        pdas (0.955308735370636)
        replacement (0.951253354549408)
        polymer (0.9139149785041809)
        lithium (0.911292552947998)
        lenmar (0.9085922241210938)
        handspring (0.9054775238037109)

sony (raw=Sony) neighbors (k=10):
        ericsson (0.9414438009262085)

dell (raw=Dell) neighbors (k=10):

***** Models:

iphone (raw=iPhone) neighbors (k=10):
        apple (0.9450328946113586)

ipad (raw=ipad) neighbors (k=10):
        ipod (0.957098126411438)
        video (0.9377815127372742)
        potato (0.9219337105751038)
        uniquesync (0.912337601184845)

thinkpad (raw=ThinkPad) neighbors (k=10):

macbook (raw=macbook) neighbors (k=10):
        macbeth (0.9610511064529419)
        aqua (0.9542038440704346)
        venom (0.9448283314704895)
        antibacterial (0.9442696571350098)
        candyshell (0.9348767399787903)
        fabshell (0.9305251240730286)
        tekkeon (0.923850417137146)
        deep (0.9232609272003174)
        kapok (0.9107840061187744)
        hurley (0.9078370928764343)

inspiron (raw=inspiron) neighbors (k=10):
        inspire (0.9423329830169678)
        hd (0.9155171513557434)

***** Attributes:

black (raw=black) neighbors (k=10):

portable (raw=portable) neighbors (k=10):

battery (raw=battery) neighbors (k=10):
        lenmar (0.9467871785163879)
        lithium (0.9336376786231995)
        replacement (0.9326379895210266)
        polymer (0.918221116065979)

***** Others:

fabshell (raw=fabshell) neighbors (k=10):
        candyshell (0.9756165146827698)
        speck (0.96562260389328)
        burton (0.9575952887535095)
        kapok (0.9550991058349609)
        teal (0.9487518668174744)
        fitted (0.9464547038078308)
        macbeth (0.9405268430709839)
        oakley (0.9396384954452515)
        antibacterial (0.9358007311820984)
        nauticase (0.9245768189430237)

hurley (raw=hurley) neighbors (k=10):
        oakley (0.98380047082901)
        harley (0.9606829285621643)
        puma (0.9446458220481873)
        kapok (0.9386962056159973)
        canopy (0.935887336730957)
        eco (0.9293815493583679)
        nauticase (0.9260969758033752)
        candyshell (0.9245275259017944)
        burton (0.9193117618560791)
        lady (0.9182815551757812)

bluetrek (raw=bluetrek) neighbors (k=10):
        blueant (0.9779480695724487)
        bluetooth (0.9718143939971924)
        enabled (0.9301869869232178)
        blu (0.9210882782936096)
        discovery (0.9179058074951172)
        headset (0.9082444310188293)

energi (raw=energi) neighbors (k=10):
        energizer (0.9870834946632385)
        to (0.9043470025062561)
```

### Integration with Search

#### week3/\_init\_.py

```python
SYNS_MODEL_LOC = os.environ.get("SYNONYMS_MODEL_LOC", "/workspace/datasets/fasttext/phone_model.bin")
print("SYNONYMS_MODEL_LOC: %s" % SYNS_MODEL_LOC)
if SYNS_MODEL_LOC and os.path.isfile(SYNS_MODEL_LOC):
    app.config["syns_model"] = fasttext.load_model(SYNS_MODEL_LOC)

    # TODO: Get the settings from an env. variable
    app.config["syns_model_stemmed"] = False
    app.config["syns_model_nn_k"] = 10

    SYNS_MODEL_KNN_THRESHOLD = os.environ.get("SYNS_MODEL_KNN_THRESHOLD", "0.90")
    app.config["syns_model_nn_threshold"] = float(SYNS_MODEL_KNN_THRESHOLD)
    print("SYNS_MODEL_KNN_THRESHOLD: %s" % SYNS_MODEL_KNN_THRESHOLD)
else:
    print("No synonym model found.  Have you run fasttext?")
app.config["index_name"] = os.environ.get("INDEX_NAME", "bbuy_annotations")
```

#### documents.py

```python
def annotate():
    debug = False
    if request.mimetype == 'application/json':
        the_doc = request.get_json()
        response = {}
        cat_model = current_app.config.get("cat_model", None) # see if we have a category model
        syns_model = current_app.config.get("syns_model", None) # see if we have a synonyms/analogies model
        # We have a map of fields to annotate.  Do POS, NER on each of them
        sku = the_doc["sku"] if "sku" in the_doc.keys() else "n/a"
        for item in the_doc:
            the_text = the_doc[item]
            if the_text is not None and the_text.find("%{") == -1:
                if item == "name":
                    if syns_model is not None:
                        if debug: print(f"[sku={sku}] IMPLEMENTED: call nearest_neighbors on your syn model and return it as `name_synonyms`")
                        # Get the nearest neighbors
                        titles_model = current_app.config["syns_model"]
                        stem = current_app.config["syns_model_stemmed"]
                        nn_k = current_app.config["syns_model_nn_k"]
                        nn_threshold = current_app.config["syns_model_nn_threshold"]
                        if stem:
                            analyzed_text = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(the_text)])
                            print(f"\t[sku={sku}] transform: {the_text} --> {analyzed_text}")
                        else:
                            analyzed_text = " ".join([token.lower() for token in tokenizer.tokenize(the_text)])

                        neighbors = titles_model.get_nearest_neighbors(analyzed_text, k=nn_k)
                        print(f"\n[sku={sku}] {analyzed_text} (raw={the_text}, neighbors (k={nn_k}):")
                        nn_list = []
                        for neighbor in neighbors:
                            neighbor_score = neighbor[0]
                            neighbor_text = neighbor[1]
                            if neighbor_score >= nn_threshold:
                                print(f"\t{neighbor_text} ({neighbor_score}) [>= threshold {nn_threshold}]")
                                nn_list.append(neighbor_text)
                            else:
                                print(f"\t{neighbor_text} ({neighbor_score})")
                        
                        syns_text = ' '.join(nn_list)
                        print(f"\tsyns_text={syns_text}")
                        response[f'{item}_synonyms'] = syns_text
        return jsonify(response)
    abort(415)
```

#### document/annotate Testing

Start the Flask-based document annotation service:

```shell
gitpod /workspace/search_with_machine_learning_course $ pyenv activate search_with_ml_week3
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ export FLASK_APP=week3
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ export SYNONYMS_MODEL_LOC=/workspace/datasets/fasttext/phone_model.bin
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ flask run --port 5000
 * Serving Flask app 'week3' (lazy loading)
 * Environment: development
 * Debug mode: on
PRIOR CLICKS: /workspace/ltr_output/train.csv
No prior clicks to load.  This may effect quality. Run ltr-end-to-end.sh per week 2 if you want
SYNONYMS_MODEL_LOC: /workspace/datasets/fasttext/phone_model.bin
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
SYNS_MODEL_KNN_THRESHOLD: 0.90
[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 435-874-441
PRIOR CLICKS: /workspace/ltr_output/train.csv
No prior clicks to load.  This may effect quality. Run ltr-end-to-end.sh per week 2 if you want
SYNONYMS_MODEL_LOC: /workspace/datasets/fasttext/phone_model.bin
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
SYNS_MODEL_KNN_THRESHOLD: 0.90
[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
```

Ad-Hoc Testing:

- q=fabshell
- Expecting: "Speck CandyShell" products

Let's verify that the "Speck CandyShell" titles have "fabshell" as a synonym:

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ grep -i CandyShell /workspace/datasets/fasttext/phone_titles.txt 
speck candyshell card case for apple iphone 4 nightdrive gray
speck candyshell case for apple iphone 3g and 3gs black
speck candyshell case for apple iphone 3g and 3gs white
speck candyshell case for apple iphone 3g and 3gs pink
speck candyshell case for apple iphone 3g and 3gs blue
speck candyshell case for apple iphone 3g and 3gs purple
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ curl -XPOST http://localhost:5000/documents/annotate -H "Content-Type:application/json" -d '{"name": "speck candyshell case for apple iphone" }'
{
  "name_synonyms": "kapok candyshell speck canopy fabshell tekkeon puma teal yellow bumper"
}
```

- q=hurley
- expecting: Hard cases like Oakley

Let's verify that the "Oakley case" titles have "hurley" as a synonym:

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ grep -i Oakley /workspace/datasets/fasttext/phone_titles.txt 
oakley o matter case for apple iphone 4 red line
oakley hazard case for apple iphone 4 black
oakley hazard case for apple iphone 4 red
oakley unobtainium case for apple iphone 4 black
oakley o matter case for apple iphone 4 black
oakley o matter case for apple iphone 4 and 4s white
oakley o matter case for apple iphone 4 and 4s sheet metal
oakley unobtainium case for apple iphone 4 and 4s red line
oakley unobtainium case for apple iphone 4 and 4s sheet metal
oakley unobtainium case for apple iphone 4 and 4s white
oakley hazard case for apple iphone 4 and 4s jet black
oakley cylinder block case for apple iphone 4 and 4s black
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ curl -XPOST http://localhost:5000/documents/annotate -H "Content-Type:application/json" -d '{"name": "oakley case for apple iphone" }'
{
  "name_synonyms": "iphone apple 4s kapok tekkeon canopy teal 4 bumper deep"
}
```

Oops, "oakley case for apple iphone" does not have "hurley" as a 10-NN neighbor!

- q=bluetrek
- expecting: bluetooth headsets like those from BlueAnt

Let's verify that the "BlueAnt bluetooth headsets" titles have "bluetrek" as a synonym:

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ curl -XPOST http://localhost:5000/documents/annotate -H "Content-Type:application/json" -d '{"name": "BlueAnt bluetooth headsets" }'
{
  "name_synonyms": "bluetooth enabled bluetrek blueant headset jabra headsets explorer"
}
```

- q=energi
- expecting: bluetooth headsets like those from BlueAnt

Let's verify that the other "Energizer chargers" titles have "energi" as a synonym:

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ curl -XPOST http://localhost:5000/documents/annotate -H "Content-Type:application/json" -d '{"name": "Energizer chargers" }'
{
  "name_synonyms": "energizer energi mycharge"
}
```



## Level 4: ...


# Self-Assessment

## For classifying product names to categories:
### What precision (P@1) were you able to achieve?

```
N       50000
P@1     0.77
R@1     0.77
```

### What fastText parameters did you use?

```shell
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/categories/categories.train -output model_categories -lr 1.0 -epoch 20 -wordNgrams 2

Read 0M words
Number of words:  15892
Number of labels: 1273
Progress: 100.0% words/sec/thread:     802 lr:  0.000000 avg.loss:  0.493649 ETA:   0h 0m 0s
```

### How did you transform the product names?

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

### How did you prune infrequent category labels, and how did that affect your precision?

I used a category min count of 10:

```
python week3/createContentTrainingData.py --output /workspace/datasets/categories/categories.fasttext --min_products 10
```

The implementation of the pruning goes as follows:

1. In a first loop, all the categories information is loaded into a Python dictionary with keys:
    - ```name```: Category's label;
    - ```count```: Number of times the category is encountered in the input;
    - ```products```: List of products' names associated with the category. 
2. In a second loop, the content of the categories dictionary is written to the output with a condition for each category that the count be greater than the passed in min counts argument, as shown below (see ```if v['count'] >= min_products```):

```python
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
```

#### Other Min Counts Experiments

All experiments were run with a 50K/50K train/test samples split. Increasing the min count from 1 to 10, 20, and 50 showed improvement in the text every time. I stopped the experiment at min count = 50.

**Min=1**

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test model_categories_min_1.bin /workspace/datasets/categories/categories_min_1.test
N       49780
P@1     0.758
R@1     0.758
```

**Min=10**

```model_categories.bin``` was trained with min count = 10.

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test model_categories.bin /workspace/datasets/categories/categories.test
N       50000
P@1     0.77
R@1     0.77
```

**Min=20**

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test model_categories_min_20.bin /workspace/datasets/categories/categories_min_20.test
N       50000
P@1     0.789
R@1     0.789
```

**Min=50**

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test model_categories_min_50.bin /workspace/datasets/categories/categories_min_50.test
N       45000
P@1     0.826
R@1     0.826
```

**Min=50, loss=hs**

Experimentation with ```loss=hs```, per this [recommendation](https://fasttext.cc/docs/en/supervised-tutorial.html#scaling-things-up).

**Train:**

Training time was significantly improved: ~1min vs. 5+ minutes for loss=softmax

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/categories/categories_min_50.train -output model_categories_min_50_loss_hs -loss hs -lr 1.0 -epoch 20 -wordNgrams 2
Read 0M words
Number of words:  14705
Number of labels: 520
Progress: 100.0% words/sec/thread:   20345 lr:  0.000000 avg.loss:  0.350199 ETA:   0h 0m 0s
```

**Test:**

Precision and Recall went down a bit. The ```loss=hs``` might be a good compromise between training speed and model's performance.

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test model_categories_min_50_loss_hs.bin /workspace/datasets/categories/categories_min_50.test
N       45000
P@1     0.791
R@1     0.791
```

### How did you prune the category tree, and how did that affect your precision?

Based on the experiments at depth 2 and 3 documented below, the precision increases at the granularity of the learned categories decreases.

#### Depth=2

**Extraction:**

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ python ./week3/createContentTrainingData.py --input ./data/pruned_products --output /workspace/datasets/categories/categories_min_50_depth_2.fasttext --min_products 50 --categories_depth 2[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
min_products=50, sample_rate=1.0, categories_depth=2
Writing labeled categories to: /workspace/datasets/categories/categories_min_50_depth_2.fasttext
Writing categories names to  : /workspace/datasets/categories/categories_names.txt
Processing pruned_products_1.xml
        19433 products processed.
Processing pruned_products_2.xml
        20786 products processed.
Processing pruned_products_3.xml
        19218 products processed.
Processing pruned_products_4.xml
        20542 products processed.
Processing pruned_products_5.xml
        19513 products processed.
Processing pruned_products_6.xml
        15866 products processed.
3 categories excluded

(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ shuf /workspace/datasets/categories/categories_min_50_depth_2.fasttext > /workspace/datasets/categories/categories_min_50_depth_2.fasttext.shuffled

(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ head -n 45000 /workspace/datasets/categories/categories_min_50_depth_2.fasttext.shuffled > /workspace/datasets/categories/categories_min_50_depth_2.train
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ tail -n 45000 /workspace/datasets/categories/categories_min_50_depth_2.fasttext.shuffled > /workspace/datasets/categories/categories_min_50_depth_2.test
```

**Training, and Testing:**

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/categories/categories_min_50_depth_2.train -output model_categories_min_50_depth_2 -lr 1.0 -epoch 20 -wordNgrams 2
Read 0M words
Number of words:  15461
Number of labels: 17
Progress: 100.0% words/sec/thread:   18757 lr:  0.000000 avg.loss:  0.052052 ETA:   0h 0m 0s

(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test model_categories_min_50_depth_2.bin /workspace/datasets/categories/categories_min_50_depth_2.test
N       45000
P@1     0.952
R@1     0.952
```

#### Depth=3

**Extraction:**

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ python ./week3/createContentTrainingData.py --input ./data/pruned_products --output /workspace/datasets/categories/categories_min_50_depth_3.fasttext --min_products 50 --categories_depth 3[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
min_products=50, sample_rate=1.0, categories_depth=3
Writing labeled categories to: /workspace/datasets/categories/categories_min_50_depth_3.fasttext
Writing categories names to  : /workspace/datasets/categories/categories_names.txt
Processing pruned_products_1.xml
        19433 products processed.
Processing pruned_products_2.xml
        20786 products processed.
Processing pruned_products_3.xml
        19218 products processed.
Processing pruned_products_4.xml
        20542 products processed.
Processing pruned_products_5.xml
        19513 products processed.
Processing pruned_products_6.xml
        15866 products processed.
93 categories excluded
```

**Training, and Testing:**

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/categories/categories_min_50_depth_3.train -output model_categories_min_50_depth_3 -lr 1.0 -epoch 20 -wordNgrams 2
Read 0M words
Number of words:  15348
Number of labels: 125
Progress: 100.0% words/sec/thread:    6168 lr:  0.000000 avg.loss:  0.089807 ETA:   0h 0m 0s

(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test model_categories_min_50_depth_3.bin /workspace/datasets/categories/categories_min_50_depth_3.test
N       45000
P@1     0.931
R@1     0.931
```

## Deriving Synonyms from Content

### What 20 tokens did you use for evaluation?

See also ```testTitleModel.py```.

The test tokens are shown below. The "others" group contains the test "synonyms" used in Level III.

```python
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
```

### What fastText parameters did you use?

```shell
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model
Read 0M words
Number of words:  2253
Number of labels: 0
Progress: 100.0% words/sec/thread:    3535 lr:  0.000000 avg.loss:  2.629022 ETA:   0h 0m 0s
```

### How did you transform the product names?

- Replace punctuation with space;
- lower case.
- The use of stemming is optional, and yielded poorer results.

```python
def transform_training_data(name):
    if stem:
        analyzed_name = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(name)])
        print(f"\ttransform: {name} --> {analyzed_name}")
    else:
        analyzed_name = " ".join([token.lower() for token in tokenizer.tokenize(name)])
    return analyzed_name
```

### What threshold score did you use?

0.90

### What synonyms did you obtain for those tokens?



## For integrating synonyms with search:
### How did you transform the product names (if different than previously)?
### What threshold score did you use?
### Were you able to find the additional results by matching synonyms?

## For classifying reviews:
### What precision (P@1) were you able to achieve?
### What fastText parameters did you use?
### How did you transform the review content?
### What else did you try and learn?

## Peer Assessment
### What are 1 or 2 things they did well in the homework?
### What are 1 or 2 concrete ways they could improve their work?
### If they indicated that they were stuck and/or want focused feedback please provide responses if you can... Feel free to add words of encouragement as well!
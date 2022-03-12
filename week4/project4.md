# Week 4 Project Report

## Level 1: Query Classification 

### Task 1: Prune the category taxonomy

#### Pruning Algorithm

See the implementation in the ```prune_categories()``` function in ```week4/create_labeled_queries.py```.

Data Structures initializations:
- The list of categories and their direct parents are loaded in ```categories``` and ```parents``` respectively from ```/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml```.
- For convenient parent-lookup, the ```parents_df``` dataframe contains one row per category in ```categories``` and its direct parent from ```parents```. Once the ```parents_df``` dataframe is constructed, the lists ```categories``` and ```parents``` are no longer needed.
- The train data in ```/workspace/datasets/train.csv``` is loaded in the data frame ```df```.
- A new column ```label``` is added, and initialized to the value of the column ```category```. The algorithm hinges on the ```label``` column as it is updated iteratively to its parent's category in the loop below as categories are rolled up (see the ```rollup``` section in the ```prune_categories``` function).
  - Note: The algorithm does _not_ change the value of the original ```category``` column, which can be useful for auditing post-mortem. At the end, the ```label``` column is used for the classification labels in the fastTest training data file that ```week4/create_labeled_queries.py``` produces.
- A new column ```audit``` is also added, and initialized to the value of the column ```category```. The ```audit``` is used to record the rollup process for any given entry in the train data, and can also be used for post-mortem auditing.

The pruning occurs by iteratively doing the following pandas-based operations of grouping, filtering, and merging:
- Update the label's parent by left-merging ```parents_df``` into ```df``` left_on=```df.label```, right_on=```parents_df.category```
- Group ```df``` by ```label``` in order to count the number of queries per label.
- Identify the labels with a number of queries < threshold via a filter operation.
- Exit the loop if there is no remaining labels that are under the threshold.
- left-merge on ```label``` the grouped and under-threshold labels into ```pd```.
- For the under-threshold labels (```row['_merge'] == 'both'```):
    - Update the label to its parent's category. **<-- This is the rollup!**
    - Update the ```audit``` column to indicate the rollup to the parent: ```new rollup > previous rollup > ... > initial category```

Illustration: Say, we start with the following records in a grouped-by-category ```df``` (Ci are categories, Pi are the parents, and the numbers are query counts; The last column is the label), and the threshold is 100:
```
P1 C1 10 Label:C1
P1 C2 20 Label:C2
P1 C3 30 Label:C3
P2 P1 40 Label:P1
```
First pass:
- 10+20+30 < 100: C1, C2, and C3 are rolled up to their parent P1.
- 30 < 100: P1 is rolled up to its parent P2.

The label column is updated as shown below:
```
P1 C1 10 Label:P1
P1 C2 20 Label:P1
P1 C3 30 Label:P1
P2 P1 40 Label:P2
```

Second pass:
- 10+20+30 < 100: The P1s are rolled up to their parents P2

The label column is updated as shown below:
```
P1 C1 10 Label:P2
P1 C2 20 Label:P2
P1 C3 30 Label:P2
P2 P1 40 Label:P2
```

Third pass: Done.
- 10+20+30+40 == 100 for rolled up P2

The pruning is done.

#### Pruning Execution

Below is the console output for the execution of ```create_labeled_queries.py``` with a ```--min_queries``` argument of 100. 7 loop iterations were necessary, leading to 880 unique categories, all with queries counts >= 100.

```shell
gitpod /workspace/search_with_machine_learning_course $ python ./week4/create_labeled_queries.py --min_queries 100 --output /workspace/datasets/labeled_query_data_min_count_100.txt
[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
min_queries=100, output_file_name=/workspace/datasets/labeled_query_data_min_count_100.txt
Processed 4639 categories from /workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml in 0:00:00.013372
Pruning /workspace/datasets/train.csv with max_loop_count=-1, query_count_threshold=100
Loading the train data from /workspace/datasets/train.csv...
WARN: No safety max loop count!
Loop 1:
        1486 unique categories
        668 labels' query counts are < 100
                Rolling up...
                Auditing...
Loop 2:
        1004 unique categories
        140 labels' query counts are < 100
                Rolling up...
                Auditing...
Loop 3:
        916 unique categories
        36 labels' query counts are < 100
                Rolling up...
                Auditing...
Loop 4:
        885 unique categories
        5 labels' query counts are < 100
                Rolling up...
                Auditing...
Loop 5:
        882 unique categories
        2 labels' query counts are < 100
                Rolling up...
                Auditing...
Loop 6:
        881 unique categories
        1 labels' query counts are < 100
                Rolling up...
                Auditing...
Loop 7:
        880 unique categories
        No label left < 100
        Ended with 880 categories
Processed 1854998 queries in 0:05:19.783830
Normalizing 1854998 queries...
... in 0:01:35.963361
Writing train data to /workspace/datasets/labeled_query_data_min_count_100.txt...
... in 0:01:39.425174
```

Below are the first 20 entries of the produced fastText train file above:
```shell
gitpod /workspace/search_with_machine_learning_course $ head -20 /workspace/datasets/fasttext/labeled_query_data_min_count_100.txt
__label__abcat0101001 television panason 50 pulgada
__label__abcat0101001 sharp
__label__pcmcat193100050014 nook
__label__abcat0101001 rca
__label__abcat0101005 rca
__label__pcmcat143200050016 flat screen tv
__label__pcmcat247400050001 macbook
__label__pcmcat171900050028 blue tooth headphon
__label__abcat0107004 tv antenna
__label__pcmcat186100050006 memori card
__label__pcmcat138100050040 ac power cord
__label__pcmcat201900050009 zagg iphon
__label__cat02713 watch the throne
__label__pcmcat224000050003 remot control extend
__label__pcmcat233600050006 camcord
__label__abcat0707001 3d
__label__abcat0410020 hoya
__label__pcmcat144700050004 wireless headphon
__label__pcmcat144700050004 wireless headphon
__label__abcat0101001 samsung 40
...
```

I produced fastTest train files for min queries of 100, 500, and 1000.

Also, for the purpose of testing the pruning algorithm, I also executed the script with a min queries of 1,000,000 and verified that the pruning ends with a single category, as shown by the console output below:

```shell
gitpod /workspace/search_with_machine_learning_course $ python ./week4/create_labeled_queries.py --min_queries 1000000 --output /workspace/datasets/labeled_query_data_min_count_1000000.txt
[nltk_data] Downloading package punkt to /home/gitpod/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
min_queries=1000000, output_file_name=/workspace/datasets/labeled_query_data_min_count_1000000.txt
Processed 4639 categories from /workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml in 0:00:00.014876
Pruning /workspace/datasets/train.csv with max_loop_count=-1, query_count_threshold=1000000
Loading the train data from /workspace/datasets/train.csv...
WARN: No safety max loop count!
Loop 1:
        1486 unique categories
        1486 labels' query counts are < 1000000
                Rolling up...
                Auditing...
Loop 2:
        369 unique categories
        369 labels' query counts are < 1000000
                Rolling up...
                Auditing...
Loop 3:
        118 unique categories
        118 labels' query counts are < 1000000
                Rolling up...
                Auditing...
Loop 4:
        43 unique categories
        42 labels' query counts are < 1000000
                Rolling up...
                Auditing...
Loop 5:
        19 unique categories
        18 labels' query counts are < 1000000
                Rolling up...
                Auditing...
Loop 6:
        7 unique categories
        6 labels' query counts are < 1000000
                Rolling up...
                Auditing...
Loop 7:
        2 unique categories
        1 labels' query counts are < 1000000
                Rolling up...
                Auditing...
Loop 8:
        1 unique categories
        No label left < 1000000
        Ended with 1 categories
Processed 1854998 queries in 0:07:34.798047
Normalizing 1854998 queries...
... in 0:01:32.910223
Writing train data to /workspace/datasets/labeled_query_data_min_count_1000000.txt...
... in 0:01:35.926158
```

### Task 2: Train a query classifier

#### fastText Train Data Preparation

##### Shuffle

```shell
gitpod /workspace/datasets $ shuf labeled_query_data_min_count_100.txt > labeled_query_data_min_count_100.txt.shuffled
gitpod /workspace/datasets $ shuf labeled_query_data_min_count_500.txt > labeled_query_data_min_count_500.txt.shuffled
gitpod /workspace/datasets $ shuf labeled_query_data_min_count_1000.txt > labeled_query_data_min_count_1000.txt.shuffled
```

##### Train & Test Samples

```shell
gitpod /workspace/search_with_machine_learning_course $ head -n 50000 /workspace/datasets/labeled_query_data_min_count_100.txt.shuffled > /workspace/datasets/labeled_query_data_min_count_100.train
gitpod /workspace/search_with_machine_learning_course $ tail -n 50000 /workspace/datasets/labeled_query_data_min_count_100.txt.shuffled > /workspace/datasets/labeled_query_data_min_count_100.test
```

#### fastText Training & Testing

##### Default fastText Parameters

**Train:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/labeled_query_data_min_count_100.train -output /workspace/datasets/labeled_query_data_min_count_100.model
Read 0M words
Number of words:  7647
Number of labels: 874
Progress: 100.0% words/sec/thread:     427 lr:  0.000000 avg.loss:  5.312009 ETA:   0h 0m 0s
```

**Test:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100.model.bin /workspace/datasets/labeled_query_data_min_count_100.test
N       49986
P@1     0.47
R@1     0.47

gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100.model.bin /workspace/datasets/labeled_query_data_min_count_100.test 5
N       49986
P@5     0.136
R@5     0.679
```

#### --epoch 10

**Train:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/labeled_query_data_min_count_100.train -output /workspace/datasets/labeled_query_data_min_count_100_epoch_20.model -epoch 10Read 0M words
Number of words:  7647
Number of labels: 874
Progress: 100.0% words/sec/thread:     410 lr:  0.000000 avg.loss:  4.029977 ETA:   0h 0m 0s
```

**Test:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100_epoch_20.model.bin /workspace/datasets/labeled_query_data_min_count_100.test
N       49986
P@1     0.507
R@1     0.507

gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100_epoch_10.model.bin /workspace/datasets/labeled_query_data_min_count_100.test 5
N       49986
P@5     0.148
R@5     0.741
```

#### --epoch 20

**Train:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/labeled_query_data_min_count_100.train -output /workspace/datasets/labeled_query_data_min_count_100_epoch_20.model -epoch 20Read 0M words
Number of words:  7647
Number of labels: 874
Progress: 100.0% words/sec/thread:     418 lr:  0.000000 avg.loss:  2.925985 ETA:   0h 0m 0s
```

**Test:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100_epoch_20.model.bin /workspace/datasets/labeled_query_data_min_count_100.test
N       49986
P@1     0.519
R@1     0.519

gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100_epoch_20.model.bin /workspace/datasets/labeled_query_data_min_count_100.test 5
N       49986
P@5     0.152
R@5     0.762
```

#### -epoch 25 -wordNgrams 2

**Train:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/labeled_query_data_min_count_100.train -output /workspace/datasets/labeled_query_data_min_count_100_epoch_25_bigrams.model -epoch 25 -wordNgrams 2
Read 0M words
Number of words:  7647
Number of labels: 874
Progress: 100.0% words/sec/thread:     409 lr:  0.000000 avg.loss:  2.680372 ETA:   0h 0m 0s
```

**Test:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100_epoch_25_bigrams.model.bin /workspace/datasets/labeled_query_data_min_count_100.test
N       49986
P@1     0.52
R@1     0.52

gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_100_epoch_25_bigrams.model.bin /workspace/datasets/labeled_query_data_min_count_100.test 5
N       49986
P@5     0.152
R@5     0.76
```

#### Min Queries 500 -epoch 25 -wordNgrams 2

**Train:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/labeled_query_data_min_count_500.train -output /workspace/datasets/labeled_query_data_min_count_500_epoch_25_bigrams.model -epoch 25 -wordNgrams 2
Read 0M words
Number of words:  7644
Number of labels: 546
Progress: 100.0% words/sec/thread:     646 lr:  0.000000 avg.loss:  2.399430 ETA:   0h 0m 0s
```

**Test:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_500_epoch_25_bigrams.model.bin /workspace/datasets/labeled_query_data_min_count_500.test
N       50000
P@1     0.526
R@1     0.526

gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_500_epoch_25_bigrams.model.bin /workspace/datasets/labeled_query_data_min_count_500.test 5
N       50000
P@5     0.154
R@5     0.768
```

#### Min Queries 1000 -epoch 25 -wordNgrams 2

**Train:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/labeled_query_data_min_count_1000.train -output /workspace/datasets/labeled_query_data_min_count_1000_epoch_25_bigrams.model -epoch 25 -wordNgrams 2
Read 0M words
Number of words:  7663
Number of labels: 388
Progress: 100.0% words/sec/thread:     716 lr:  0.000000 avg.loss:  2.167381 ETA:   0h 0m 0s
```

**Test:**

```shell
gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_1000_epoch_25_bigrams.model.bin /workspace/datasets/labeled_query_data_min_count_1000.test
N       50000
P@1     0.527
R@1     0.527

gitpod /workspace/search_with_machine_learning_course $ ~/fastText-0.9.2/fasttext test /workspace/datasets/labeled_query_data_min_count_1000_epoch_25_bigrams.model.bin /workspace/datasets/labeled_query_data_min_count_1000.test 5
N       50000
P@5     0.154
R@5     0.772 <-- THE BEST IN MY EXPERIMENTS
```

#### TODO: Experiment with different text analysis in create_labeled_queries.py

## Level 2: 

### Products & Queries Ingestion

The Kaggle Best Buy products and Queries datasets were ingested from scratch using the week4/conf OpenSearch mappings and the logstash config used in week 1 or 2.

```
GET /bbuy_products/_count

Result:
{
  "count" : 1275077,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  }
}

GET /bbuy_queries/_count

Result:
{
  "count" : 1865269,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  }
}
```

### Task 1: Add the query classifier to query processing

#### Load your fastText model

Note: All fastText models produced in Level 1 were moved to the /workspace/datasets/fasttext directory in order to not clutter /workspace/datasets.

The QUERY_CLASS_MODEL_LOC environment variable was set to the best model I produced in the previous level, and verified that the Flask application loaded it successfully:

```shell
export QUERY_CLASS_MODEL_LOC=/workspace/datasets/fasttext/labeled_query_data_min_count_1000_epoch_25_bigrams.model.bin

(search_with_ml_week4) gitpod /workspace/search_with_machine_learning_course $ flask run --port 3000
 * Serving Flask app 'week4' (lazy loading)
 * Environment: development
 * Debug mode: on
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
QUERY_CLASS_MODEL_LOC: /workspace/datasets/fasttext/labeled_query_data_min_count_1000_epoch_25_bigrams.model.bi
...
```

#### Classify the query into a category using your model

```python
def get_query_category(user_query, query_class_model, debug = False):
    if debug: print("IMPLEMENTED: get_query_category")
    assert query_class_model is not None
    predictions = query_class_model.predict(user_query, k = 10)
    assert predictions is not None
    assert len(predictions) == 2

    # Accumulate the predicted categories
    classifications = []
    classifications_confidence_accumulated = 0.0
    classifications_confidence_accumulated_min = current_app.config["classifications_confidence_accumulated_min"]
    classification_confidence_min = current_app.config["classification_confidence_min"]
    for i, classification in enumerate(predictions[0]):
        print(f"[{i}] classification={predictions[0][i]}, with probability {predictions[1][i]}")
        if classifications_confidence_accumulated < classifications_confidence_accumulated_min:
            if predictions[1][i] >= classification_confidence_min:
                classifications_confidence_accumulated += predictions[1][i]
                classifications.append(predictions[0][i][9:]) # [9:] removes the "__label__" prefix
                print(f"\tAcc. confidence: {classifications_confidence_accumulated}")
            else:
                print(f"\tConfidence: {predictions[1][i]} is too low (threshold={classification_confidence_min})")
        else:
            print(f"Reached targeted min accumulated confidence {classifications_confidence_accumulated_min}")
    if debug: print(f"Returning: {classifications}")
    return classifications
```

In ```search.py::query()```, a category filter is added when predictions are available (```if query_category is not None and len(query_category) > 0```):

```python
    query_class_model = current_app.config["query_model"]
    query_category = get_query_category(user_query, query_class_model, debug = DEBUG)
    if query_category is not None and len(query_category) > 0:
        if DEBUG: print("IMPLEMENTED: add this into the filters object so that it gets applied at search time.  This should look like your `term` filter from week 1 for department but for categories instead")
        query_obj['query']['bool']['filter'].append(
            {
                'terms': {
                    'categoryPathIds.keyword': query_category
                }
            }
        )
        if DEBUG:
            print(f"Added category classification filter: {query_category}")
            print(f"Updated Query obj with category filter: {json.dumps(query_obj, indent = 4)}")
```

#### Observations

**Good:**

- The recall is significantly reduced
- The relevance (with my eyes) seems to have increased across sample queries.
- Sorting by a factor other than Relevance (or by Relevance ascending) provides a way to visit the "bottom" of the results set and observe whether the results are still somewhat relevant thanks to the category filter.

**Bad:**

Some queries did not return any result with the category filter:

- q=laptops
- 

### Task 2: Use the query classifier output to filter results

### Task 3: Use the query classifier output to boost results (Optional


## Project Assessment

To assess your project work, you should be able to answer the following questions:

### For query classification:

- How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 100? To 1000?

- What values did you achieve for P@1, R@3, and R@5? You should have tried at least a few different models, varying the minimum number of queries per category as well as trying different fastText parameters or query normalization. Report at least 3 of your runs.

### For integrating query classification with search:

- Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.

- Given 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.
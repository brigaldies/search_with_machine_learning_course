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

### Task 2: Use the query classifier output to filter results

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

- As expected, the recall is significantly reduced.
- The relevance (with my eyes) seems to have increased across sample queries.
- Sorting by a factor other than Relevance (or by Relevance ascending) provides a way to visit the "bottom" of the results set and observe whether the results are still somewhat relevant thanks to the category filter.

Examples of queries:

**q=ipad**

**Without Query Classification Filter**

![q=ipad, without Query Classification Filter](./images/q%3Dipad_no_query_classification.png)

**With Query Classification Filter**

![q=ipad, with Query Classification Filter](./images/q%3Dipad_with_query_classification_filter.png)

**q=dell laptops**

**Without Query Classification Filter**

![q=dell laptops, without Query Classification Filter](./images/q%3Ddell_laptops_no_query_classification.png)

**With Query Classification Filter**

![q=dell laptops, with Query Classification Filter](./images/q%3Ddell_laptops_with_query_classification_filter.png)

**Bad:**

Some seemingly easy to predict queries did not classify properly:

- q=laptops: The classification to "Movies & TV Shows [cat02015; conf=0.41]: Best Buy [cat00000] > Movies & Music [abcat0600000] > Movies & TV Shows [cat02015]" is just horrible!

### Task 3: Use the query classifier output to boost results (Optional)

Additional environment variables were introduced in order to enable/disable query classification, and choose between filtering and boosting:

```python
# Default: Enabled
query_classification_enabled = True if os.environ.get("QUERY_CLASS_ENABLED", "true").lower() == "true" else False
app.config["query_classification_enabled"] = query_classification_enabled
print(f"query_classification_enabled={query_classification_enabled}")

# Default: Do not use the filter; Boost instead
query_classification_as_filter = True if os.environ.get("QUERY_CLASS_AS_FILTER", "false").lower() == "true" else False
app.config["query_classification_as_filter"] = query_classification_as_filter
print(f"query_classification_as_filter={query_classification_as_filter}")

# Default boost value: 1000
query_classification_boost = os.environ.get("QUERY_CLASS_BOOST", 1000)
app.config["query_classification_boost"] = query_classification_boost
print(f"query_classification_boost={query_classification_boost}")
```

In ```search.py::query()```, in case of boosting, an additional should clause is added to boost on the predicated categories, as shown below:

```python
    if current_app.config["query_classification_enabled"]:
        query_class_model = current_app.config["query_model"]
        query_category = get_query_category(user_query, query_class_model, debug = DEBUG)
        if query_category is not None and len(query_category) > 0:
            if DEBUG: print("IMPLEMENTED: add this into the filters object so that it gets applied at search time.  This should look like your `term` filter from week 1 for department but for categories instead")
            predicted_categories_clause = {
                        'terms': {
                            'categoryPathIds.keyword': [category[0] for category in query_category]
                        }
                    }
            if current_app.config["query_classification_as_filter"]:
                if 'filter' in query_obj['query']['bool'].keys() and query_obj['query']['bool']['filter'] is not None:
                    query_obj['query']['bool']['filter'].append(predicted_categories_clause)
                else:
                    query_obj['query']['bool']['filter'] = predicted_categories_clause
                if DEBUG:
                    print(f"Added category filter: {query_category}")
            else:
                predicted_categories_clause['terms']['boost'] = current_app.config["query_classification_boost"]
                if 'bool' in query_obj['query']:
                    if 'should' in query_obj['query']['bool'].keys() and query_obj['query']['bool']['should'] is not None:
                        query_obj['query']['bool']['should'].append(predicted_categories_clause)
                    else:
                        query_obj['query']['bool']['should'] = predicted_categories_clause
                    if DEBUG:
                        print(f"Added category boost: {query_category}")
                else:
                    print(f"WARN: Did not find the expected boolean query.")
            print(f"Updated Query obj with category filter or boost: {json.dumps(query_obj, indent = 4)}")

            # Lookup the predicated categories' names and paths (to be displayed at the top of the page)
            predicted_categories = []
            categories_df = current_app.config["categories_df"]
            predicted_categories_df = categories_df[categories_df['id'].isin([category[0] for category in query_category])]
            for index, row in predicted_categories_df.iterrows():
                confidence = 'N/A'
                for prediction in query_category:
                    if prediction[0] == row['id']:
                        confidence = prediction[1]
                        break

                predicted_categories.append(f"{row['name']} [{row['id']}; conf={confidence:.2f}]: {row['path']}")
    else:
        print(f"WARN: Query classification is disabled")
        predicted_categories = []
```

#### Observations

**Good:**

- I had to boost with a x1000 multiplier to start seeing the results associated with the predicted categories.
- In general, the technique seems effective.

TODO: Add some screen shots.

**Bad:**

- See Self-assessment.

## Project Assessment

To assess your project work, you should be able to answer the following questions:

### For query classification:

- How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 100? To 1000?

I ran ./week4/create_labeled_queries.py for min queries 100, 500, 1000, and 1000000 (to test that the rollup algorithm ends at one category left), which generated the following unique categories counts:

```
100    : 880
500    : 546
1000   : 300
1000000: 1
```

- What values did you achieve for P@1, R@3, and R@5? You should have tried at least a few different models, varying the minimum number of queries per category as well as trying different fastText parameters or query normalization. Report at least 3 of your runs.

|Min Queries|epochs|wordNgrams|P@1|R@1|P@5|R@5|Comment|
|----------:|-----:|----:|--:|--:|--:|--:|--|
|100        |5    |-|0.47 |0.47|0.136|0.679||
|100        |10   |-|0.507|0.507|0.148|0.741||
|100        |20   |-|0.519|0.519|0.152|0.762||
|100        |25   |2|0.52|0.52|0.152|0.76|Slight decline of R@5|
|500        |25   |2|0.526|0.526|0.154|0.768||
|1000       |25   |2|0.527|0.527|0.154|0.772|THE BEST RESULT IN MY EXPERIMENTS|

### For integrating query classification with search:

- Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.

**q=ipad**

See screen shots without and with the category(ies) filter earlier in the report.

**Query classification** - See the ```get_query_category``` output below:

```
[0] classification=__label__pcmcat209000050007, with probability 0.6183217763900757
        Acc. confidence: 0.6183217763900757
[1] classification=__label__pcmcat209000050008, with probability 0.12028033286333084
        Acc. confidence: 0.7386021092534065
[2] classification=__label__pcmcat218000050000, with probability 0.03238596394658089
        Confidence: 0.03238596394658089 is too low (threshold=0.1)
[3] classification=__label__pcmcat217900050000, with probability 0.03223486989736557
        Confidence: 0.03223486989736557 is too low (threshold=0.1)
[4] classification=__label__pcmcat218000050003, with probability 0.02927657775580883
        Confidence: 0.02927657775580883 is too low (threshold=0.1)
[5] classification=__label__pcmcat144700050004, with probability 0.016998620703816414
        Confidence: 0.016998620703816414 is too low (threshold=0.1)
[6] classification=__label__abcat0208011, with probability 0.01401718333363533
        Confidence: 0.01401718333363533 is too low (threshold=0.1)
[7] classification=__label__pcmcat218000050002, with probability 0.01293209008872509
        Confidence: 0.01293209008872509 is too low (threshold=0.1)
[8] classification=__label__pcmcat193100050014, with probability 0.010597367770969868
        Confidence: 0.010597367770969868 is too low (threshold=0.1)
[9] classification=__label__pcmcat171900050024, with probability 0.008955840952694416
        Confidence: 0.008955840952694416 is too low (threshold=0.1)
Returning: [('pcmcat209000050007', 0.6183217763900757), ('pcmcat209000050008', 0.12028033286333084)]
```

**Category(ies) filter displayed at the top of the page:**

```
iPad [pcmcat209000050007; conf=0.62]: Best Buy [cat00000] > Computers & Tablets [abcat0500000] > Tablets & iPad [pcmcat209000050006] > iPad [pcmcat209000050007]

Tablets [pcmcat209000050008; conf=0.12]: Best Buy [cat00000] > Computers & Tablets [abcat0500000] > Tablets & iPad [pcmcat209000050006] > Tablets [pcmcat209000050008]
```

**q=dell laptops**

**Query classification** - See the ```get_query_category``` output below:

```
[0] classification=__label__pcmcat247400050000, with probability 0.8775404095649719
        Acc. confidence: 0.8775404095649719
[1] classification=__label__pcmcat212600050008, with probability 0.03798757493495941
        Confidence: 0.03798757493495941 is too low (threshold=0.1)
[2] classification=__label__pcmcat164200050013, with probability 0.03649915009737015
        Confidence: 0.03649915009737015 is too low (threshold=0.1)
[3] classification=__label__pcmcat183800050006, with probability 0.012372598983347416
        Confidence: 0.012372598983347416 is too low (threshold=0.1)
[4] classification=__label__pcmcat183800050007, with probability 0.005114967469125986
        Confidence: 0.005114967469125986 is too low (threshold=0.1)
[5] classification=__label__pcmcat190000050014, with probability 0.0043478114530444145
        Confidence: 0.0043478114530444145 is too low (threshold=0.1)
[6] classification=__label__pcmcat219300050014, with probability 0.0038257346022874117
        Confidence: 0.0038257346022874117 is too low (threshold=0.1)
[7] classification=__label__pcmcat209000050008, with probability 0.003420345252379775
        Confidence: 0.003420345252379775 is too low (threshold=0.1)
[8] classification=__label__abcat0811004, with probability 0.003255515592172742
        Confidence: 0.003255515592172742 is too low (threshold=0.1)
[9] classification=__label__pcmcat247400050001, with probability 0.002136114053428173
        Confidence: 0.002136114053428173 is too low (threshold=0.1)
Returning: [('pcmcat247400050000', 0.8775404095649719)]
```

**Category(ies) filter displayed at the top of the page:**
```
PC Laptops [pcmcat247400050000; conf=0.88]: Best Buy [cat00000] > Computers & Tablets [abcat0500000] > Laptop & Netbook Computers [abcat0502000] > PC Laptops [pcmcat247400050000]
```

**q=printer ink**

**Query classification** - See the ```get_query_category``` output below:

```
[0] classification=__label__abcat0807005, with probability 0.7549535632133484
        Acc. confidence: 0.7549535632133484
[1] classification=__label__abcat0807001, with probability 0.12224579602479935
        Acc. confidence: 0.8771993592381477
[2] classification=__label__abcat0511004, with probability 0.050132088363170624
        Confidence: 0.050132088363170624 is too low (threshold=0.1)
[3] classification=__label__abcat0511002, with probability 0.02771952375769615
        Confidence: 0.02771952375769615 is too low (threshold=0.1)
[4] classification=__label__abcat0511001, with probability 0.014386196620762348
        Confidence: 0.014386196620762348 is too low (threshold=0.1)
[5] classification=__label__abcat0400000, with probability 0.008122657425701618
        Confidence: 0.008122657425701618 is too low (threshold=0.1)
[6] classification=__label__pcmcat245100050028, with probability 0.006479900795966387
        Confidence: 0.006479900795966387 is too low (threshold=0.1)
[7] classification=__label__abcat0511007, with probability 0.00522206025198102
        Confidence: 0.00522206025198102 is too low (threshold=0.1)
[8] classification=__label__pcmcat172000050000, with probability 0.0027377076912671328
        Confidence: 0.0027377076912671328 is too low (threshold=0.1)
[9] classification=__label__abcat0515013, with probability 0.0018335168715566397
        Confidence: 0.0018335168715566397 is too low (threshold=0.1)
Returning: [('abcat0807005', 0.7549535632133484), ('abcat0807001', 0.12224579602479935)]
```

**Category(ies) filter displayed at the top of the page:**

```
Printer Ink [abcat0807001; conf=0.12]: Best Buy [cat00000] > Office [pcmcat245100050028] > Printer Ink & Toner [abcat0807000] > Printer Ink [abcat0807001]

Hewlett-Packard [abcat0807005; conf=0.75]: Best Buy [cat00000] > Office [pcmcat245100050028] > Printer Ink & Toner [abcat0807000] > Printer Ink [abcat0807001] > Hewlett-Packard [abcat0807005]
```

- Given 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.

**No Classification With Confidence > 0.5**

**q=laptops**

The classifier was unable to classify with a minimum of 0.5 confidence:

```
[0] classification=__label__cat02015, with probability 0.41393041610717773
        Acc. confidence: 0.41393041610717773
[1] classification=__label__cat09000, with probability 0.07070089876651764
        Confidence: 0.07070089876651764 is too low (threshold=0.1)
[2] classification=__label__cat02009, with probability 0.04854829981923103
        Confidence: 0.04854829981923103 is too low (threshold=0.1)
[3] classification=__label__pcmcat247400050000, with probability 0.02866331674158573
        Confidence: 0.02866331674158573 is too low (threshold=0.1)
[4] classification=__label__abcat0101001, with probability 0.02822437696158886
        Confidence: 0.02822437696158886 is too low (threshold=0.1)
[5] classification=__label__pcmcat209400050001, with probability 0.016706274822354317
        Confidence: 0.016706274822354317 is too low (threshold=0.1)
[6] classification=__label__cat02006, with probability 0.013782092370092869
        Confidence: 0.013782092370092869 is too low (threshold=0.1)
[7] classification=__label__pcmcat242800050021, with probability 0.01290363259613514
        Confidence: 0.01290363259613514 is too low (threshold=0.1)
[8] classification=__label__abcat0301014, with probability 0.00840129517018795
        Confidence: 0.00840129517018795 is too low (threshold=0.1)
[9] classification=__label__pcmcat144700050004, with probability 0.008179730735719204
        Confidence: 0.008179730735719204 is too low (threshold=0.1)
Returning: []
No query classification available!
```

Same issue (no classification with a confidence > 0.5) for the following queries:
- apple ipad
- macbooks
- computer keywords
- wifi routers
- network cables

**Wrong Classification**

**q=monitors**

**Query classification** - See the ```get_query_category``` output below:

```
[0] classification=__label__cat02015, with probability 0.5299343466758728
        Acc. confidence: 0.5299343466758728
[1] classification=__label__abcat0101001, with probability 0.08168108016252518
        Confidence: 0.08168108016252518 is too low (threshold=0.1)
[2] classification=__label__cat09000, with probability 0.07261267304420471
        Confidence: 0.07261267304420471 is too low (threshold=0.1)
[3] classification=__label__pcmcat247400050000, with probability 0.038371406495571136
        Confidence: 0.038371406495571136 is too low (threshold=0.1)
[4] classification=__label__cat02009, with probability 0.02427739091217518
        Confidence: 0.02427739091217518 is too low (threshold=0.1)
[5] classification=__label__cat02010, with probability 0.011420287191867828
        Confidence: 0.011420287191867828 is too low (threshold=0.1)
[6] classification=__label__pcmcat144700050004, with probability 0.010543576441705227
        Confidence: 0.010543576441705227 is too low (threshold=0.1)
[7] classification=__label__pcmcat209400050001, with probability 0.01016100775450468
        Confidence: 0.01016100775450468 is too low (threshold=0.1)
[8] classification=__label__pcmcat242800050021, with probability 0.008323170244693756
        Confidence: 0.008323170244693756 is too low (threshold=0.1)
[9] classification=__label__abcat0900000, with probability 0.007755163125693798
        Confidence: 0.007755163125693798 is too low (threshold=0.1)
Returning: [('cat02015', 0.5299343466758728)]
```

**Category(ies) filter displayed at the top of the page:**

```
Movies & TV Shows [cat02015; conf=0.53]: Best Buy [cat00000] > Movies & Music [abcat0600000] > Movies & TV Shows [cat02015]
```
# Week 4 Project Report

## Level 1: Query Classification 

### Pruning Algorithm

See the implementation in the ```prune_categories``` function in ```week4/create_labeled_queries.py```.

Data Structures initializations:
- The list of categories and their direct parents are loaded in ```categories``` and ```parents``` respectively from ```/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml```.
- For convenient parent-lookup, the ```parents_df``` dataframe contains one row per category in ```categories``` and its direct parent from ```parents```. Once the ```parents_df``` dataframe is constructed, the lists ```categories``` and ```parents``` are no longer needed.
- The train data in ```/workspace/datasets/train.csv``` is loaded in the data frame ```df```.
- A new column ```label``` is added, and initialized to the value of the column ```category```. The algorithm hinges on the ```label``` column as it is updated iteratively to its parent's category in the loop below as categories are rolled up (see the ```rollup``` section in the ```prune_categories``` function). The algorith does _not_ change the value of the original ```category``` column, which can be useful for auditing. At the end, the ```label``` column is used for the labels in the fastTest training data file that ``week4/create_labeled_queries.py``` produces.
- A new column ```audit``` is added, and initialized to the value of the column ```category```. The ```audit``` is used to record the rollup process for any given entry in the train data.

The pruning occurs by iteratively doing the following pandas-based operations of grouping, filtering, and merging:
- Update the label's parent by left-merging ```parents_df``` into ```df``` left_on=```df.label```, right_on=```parents_df.category```
- Group ```df``` by ```label``` in order to count the number of queries per label.
- Identify the labels with a number of queries < threshold via a filter operation.
- Exit the loop if there is no remaining labels that are under the threshold.
- left-merge on ```label``` the grouped and under-threshold labels into ```pd```.
- For the under-threshold labels (```row['_merge'] == 'both'```):
    - update the label to its parent's category.
    - Update the ```audit``` column to indicate the rollup to the parent: ```new rollup > previous rollup > ... > initial category```

### Task 1: Prune the category taxonomy


### Task 2: Train a query classifier



## Level 2: 

### Task 1: Add the query classifier to query processing

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
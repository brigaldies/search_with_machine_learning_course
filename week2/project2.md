# Week 2 - Project 2

## Level 1: Implementing End to End LTR for our Dataset

### Step 1: Be sure you’ve

1. Setup the project per above (re-indexed content, etc.)
    - Done.
1. Your week2 pyenv is activated: `pyenv activate search_with_ml_week2`
    - Done.

### Step 2: Familiarize yourself with the code and our query logs

- `./week2/utilities`
    - Done.
- `/ltr-end-to-end.sh`
    - Done.

### Step 3: Try out a few commands from the terminal in your Gitpod UI

1. Run `python week2/utilities/build_ltr.py -h`
    - Done.
1. Run `python week2/utilities/build_ltr.py --lookup_query "canon rebel" --all_clicks /workspace/datasets/train.csv | less `
    - Done.
1. Run `python week2/utilities/build_ltr.py --lookup_product 1980124`
    - Done.
1. Familiarize yourself with Pandas
    - Done.
1. Execute `mkdir /workspace/ltr_output` and then `cp data/validity.csv /workspace/ltr_output/` 

```validity.csv``` was produced with `build_ltr –verify_products …`, and identifies skus from the training data which do NOT exist in the index.

1. Run `./ltr-end-to-end.sh -y`

Reviewed code for:
- `python week2/utilities/build_ltr.py --create_ltr_store`
- `python week2/utilities/build_ltr.py -f week2/conf/ltr_featureset.json --upload_featureset`
- `python week2/utilities/build_ltr.py --output_dir /workspace/ltr_output --split_input /workspace/datasets/train.csv --split_train_rows 1000000 --split_test_rows 1000000`
- `python week2/utilities/build_ltr.py --generate_impressions --output_dir /workspace/ltr_output --train_file /workspace/ltr_output/train.csv --synthesize` *** TODO: Review again "rank" and "impressions calculations ***
- `python week2/utilities/build_ltr.py --ltr_terms_field sku --output_dir /workspace/ltr_output --create_xgb_training -f week2/conf/ltr_featureset.json --click_model heuristic`
- `python week2/utilities/build_ltr.py  --output_dir /workspace/ltr_output -xgb /workspace/ltr_output/training.xgb --xgb_conf week2/conf/xgb-conf.json`
- `python week2/utilities/build_ltr.py --upload_ltr_model --xgb_model /workspace/ltr_output/xgb_model.model`
- `python week2/utilities/build_ltr.py --xgb_plot --output_dir /workspace/ltr_output`
- `python week2/utilities/build_ltr.py --xgb_test /workspace/ltr_output/test.csv --train_file /workspace/ltr_output/train.csv --output_dir /workspace/ltr_output --xgb_test_num_queries 200`
- `python week2/utilities/build_ltr.py --analyze --output_dir /workspace/ltr_output`

Run #1:

(search_with_ml_week2) gitpod /workspace/search_with_machine_learning_course $ python week2/utilities/build_ltr.py --analyze --output_dir /workspace/ltr_output
Analyzing results from /workspace/ltr_output/xgb_test_output.csv
Queries not seen during training: [175]
                 query
0              labtops
1             speakers
2             scream 4
3    laptop hard drive
4         iPhone cases
..                 ...
170          laminator
171            ds lite
172            Hp envy
173             router
174             Kindle

[175 rows x 1 columns]


Simple MRR is 0.362
LTR Simple MRR is 0.355
Hand tuned MRR is 0.428
LTR Hand Tuned MRR is 0.410

Simple p@10 is 0.139
LTR simple p@10 is 0.138
Hand tuned p@10 is 0.203
LTR hand tuned p@10 is 0.188
Simple better: 625      LTR_Simple Better: 483  Equal: 1887
HT better: 1573 LTR_HT Better: 801      Equal: 963
Saving Better/Equal analysis to /workspace/ltr_output/analysis

Run #2:

(search_with_ml_week2) gitpod /workspace/search_with_machine_learning_course $ python week2/utilities/build_ltr.py --analyze --output_dir /workspace/ltr_output
Analyzing results from /workspace/ltr_output/xgb_test_output.csv
Analyzing results from /workspace/ltr_output/xgb_test_output.csv
Queries not seen during training: [178]
               query
0    bobby valentino
1     computer desks
2        cruz tablet
3                Dre
4      ethernet card
..               ...
173        star trek
174         Apple tv
175    samsung dryer
176           kindel
177     Beats Studio

[178 rows x 1 columns]


Simple MRR is 0.351
LTR Simple MRR is 0.332
Hand tuned MRR is 0.423
LTR Hand Tuned MRR is 0.403

Simple p@10 is 0.123
LTR simple p@10 is 0.122
Hand tuned p@10 is 0.194
LTR hand tuned p@10 is 0.179
Simple better: 349      LTR_Simple Better: 261  Equal: 1401
HT better: 1131 LTR_HT Better: 475      Equal: 696
Saving Better/Equal analysis to /workspace/ltr_output/analysis

## Level 2: Exploring Features and Click Models
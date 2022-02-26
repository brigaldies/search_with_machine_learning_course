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
- `python week2/utilities/build_ltr.py  --output_dir /workspace/ltr_output --xgb /workspace/ltr_output/training.xgb --xgb_conf week2/conf/xgb-conf.json`
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

Run #3: After re-implemented click_models.step()

(search_with_ml_week2) gitpod /workspace/search_with_machine_learning_course $ python week2/utilities/build_ltr.py --analyze --output_dir /workspace/ltr_output
Analyzing results from /workspace/ltr_output/xgb_test_output.csv
Queries not seen during training: [180]
                        query
0                     windows
1                  ipod video
2                  blue tooth
3                      remote
4                         jvc
..                        ...
175              speaker wire
176         computer monitors
177            evo hdmi cable
178       my chemical romance
179  How to train your dragon

[180 rows x 1 columns]


Simple MRR is 0.352
LTR Simple MRR is 0.348
Hand tuned MRR is 0.435
LTR Hand Tuned MRR is 0.432

Simple p@10 is 0.136
LTR simple p@10 is 0.135
Hand tuned p@10 is 0.192
LTR hand tuned p@10 is 0.184
Simple better: 573      LTR_Simple Better: 495  Equal: 1699
HT better: 1428 LTR_HT Better: 864      Equal: 669
Saving Better/Equal analysis to /workspace/ltr_output/analysis

Run #4: With Down Sampling

Analyzing results from /workspace/ltr_output/xgb_test_output.csv
Queries not seen during training: [178]
                   query
0    across the universe
1                   kids
2              Alienware
3         computer games
4      Otterbox iPhone 4
..                   ...
173       car subwoofers
174            backpacks
175           iPod touch
176      wireless router
177          Car in dash

[178 rows x 1 columns]


Simple MRR is 0.348
LTR Simple MRR is 0.336
Hand tuned MRR is 0.408
LTR Hand Tuned MRR is 0.402

Simple p@10 is 0.133
LTR simple p@10 is 0.133
Hand tuned p@10 is 0.179
LTR hand tuned p@10 is 0.170
Simple better: 443      LTR_Simple Better: 383  Equal: 1638
HT better: 1202 LTR_HT Better: 791      Equal: 578
Saving Better/Equal analysis to /workspace/ltr_output/analysis

## Level 2: Exploring Features and Click Models

## Self Assessment

Project Self Assessment
1. Do you understand the steps involved in creating and deploying an LTR model? Name them and describe what each step does in your own words.
2. What is a feature and featureset?
3. What is the difference between precision and recall?
4. What are some of the traps associated with using click data in your model?
5. What are some of the ways we are faking our data and how would you prevent that in your application?
6. What is target leakage and why is it a bad thing?
7. When can using prior history cause problems in search and LTR?
8. Submit your project along with your best MRR scores
Peer Review Questions
1. What are 1 or 2 things they did well in the homework?
2. What are 1 or 2 concrete ways they could improve their work?
3. If they indicated that they were stuck and/or want focused feedback please provide responses if you can...
4. Feel free to add words of encouragement as well!
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







## Level 2: Exploring Features and Click Models
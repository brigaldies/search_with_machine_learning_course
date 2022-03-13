import os

from flask import Flask
from flask import render_template

import fasttext
import pandas as pd

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
        QUERY_CLASS_MODEL_LOC = os.environ.get("QUERY_CLASS_MODEL_LOC", "/workspace/datasets/fasttext/query_model.bin")
        if QUERY_CLASS_MODEL_LOC and os.path.isfile(QUERY_CLASS_MODEL_LOC):
            app.config["query_model"] = fasttext.load_model(QUERY_CLASS_MODEL_LOC)
        else:
            print("No query model found.  Have you run fasttext?")
        print("QUERY_CLASS_MODEL_LOC: %s" % QUERY_CLASS_MODEL_LOC)

        app.config["classifications_confidence_accumulated_min"] = os.environ.get("QUERY_CLASS_ACC_CONFIDENCE_MIN", 0.5)
        print(f"classifications_confidence_accumulated_min={app.config['classifications_confidence_accumulated_min']}")

        app.config["classification_confidence_min"] = os.environ.get("QUERY_CLASS_CONFIDENCE_MIN", 0.1)
        print(f"classification_confidence_min={app.config['classification_confidence_min']}")

        categories_csv = os.environ.get("CATEGORIES", "/workspace/datasets/product_data/categories/categories.csv")
        cat_df = pd.read_csv(categories_csv, sep='\t')
        app.config["categories_df"] = cat_df
        print(f"Loaded {cat_df.shape[0]} categories from {categories_csv}")

        query_classification_enabled = True if os.environ.get("QUERY_CLASS_ENABLED", "true").lower() == "true" else False
        # Override
        # query_classification_enabled = False
        app.config["query_classification_enabled"] = query_classification_enabled
        print(f"query_classification_enabled={query_classification_enabled}")

        query_classification_as_filter = True if os.environ.get("QUERY_CLASS_AS_FILTER", "false").lower() == "true" else False
        app.config["query_classification_as_filter"] = query_classification_as_filter
        print(f"query_classification_as_filter={query_classification_as_filter}")

        query_classification_boost = os.environ.get("QUERY_CLASS_BOOST", 1000)
        app.config["query_classification_boost"] = query_classification_boost
        print(f"query_classification_boost={query_classification_boost}")
        
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    app.config["index_name"] = os.environ.get("INDEX_NAME", "bbuy_products")
    
    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # A simple landing page
    #@app.route('/')
    #def index():
    #    return render_template('index.jinja2')

    from . import search
    app.register_blueprint(search.bp)
    app.add_url_rule('/', view_func=search.query)

    return app

import json
import sys
import tempfile
from urllib.parse import urljoin

import requests
import xgboost as xgb
from opensearchpy import OpenSearch
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_tree
import pandas as pd

impressions_df = pd.read_csv("/workspace/ltr_output/impressions.csv")
query_gb = impressions_df.groupby("query")
key = 'lcd tv'
group = query_gb.get_group(key)
doc_ids = group.doc_id.values
doc_ids = doc_ids.tolist()

def create_feature_log_query(query, doc_ids, click_prior_query, featureset_name, ltr_store_name, size=200, terms_field="_id", debug = False):
    if debug: print(f"IMPLEMENTED: create_feature_log_query for query={query}, doc_ids={doc_ids}, click_prior_query={click_prior_query}, featureset={featureset_name}, store={ltr_store_name}, size={size}, terms_field={terms_field}")
    features_logging_query = {
        "size": size,
        "_source": [ "name", "sku" ],
        "query": {
            "bool": {
                "filter": [
                    {
                        # Doc ids to retrieve LTR features for
                        "terms": {
                            terms_field: doc_ids
                        }
                    },
                    {
                    "sltr": {
                        "_name": "logged_featureset",
                        "featureset": featureset_name,
                        "store": ltr_store_name,
                        "params": {
                            # The query for the query-dependent features
                            "keywords": query
                        }
                    }
                    }
                ]
                }
        },
        # Return the LTR features
        "ext": {
            "ltr_log": {
                "log_specs": {
                    "name": "log_entry",
                    "named_query": "logged_featureset"
                }
            }
        }
    }
    if debug: print(f"Features logging query: {features_logging_query}")
    return features_logging_query

host = 'localhost'
port = 9200
base_url = "https://{}:{}/".format(host, port)
auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.

# Create the client with SSL/TLS enabled, but hostname and certificate verification disabled.
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_compress=True,  # enables gzip compression for request bodies
    http_auth=auth,
    # client_cert = client_cert_path,
    # client_key = client_key_path,
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

def __log_ltr_query_features(query_id, key, query_doc_ids, click_prior_query, no_results, terms_field="_id"):
        debug = True
        featureset_name = 'bbuy_main_featureset'
        ltr_store_name = 'week2'
        index_name = 'bbuy_products'
        log_query = create_feature_log_query(key, query_doc_ids, click_prior_query, featureset_name,
                                                ltr_store_name,
                                                size=len(query_doc_ids), terms_field=terms_field, debug = debug)
                                                # Run the query just like any other search
        response = client.search(body=log_query, index=index_name)

        # New implementation
        docs_with_features_list = []
        returned_doc_ids = []
        for hit in response['hits']['hits']:
            returned_doc_ids.append(hit['_id'])
            row_dict = {
                'doc_id': hit['_id'],
                'query_id': query_id.iloc[0],
                'sku': hit['_source']['sku'][0]
            }
            for feature in hit['fields']['_ltrlog'][0]['log_entry']:
                row_dict[feature.get('name')] = feature.get('value', 0)
            docs_with_features_list.append(row_dict)

        frame = pd.DataFrame(docs_with_features_list)
        return frame.astype({'doc_id': 'int64', 'query_id': 'int64', 'sku': 'int64'})
        # return frame
        # return docs_with_features_list

no_results = {}
click_prior_query = 'ignored for now'
terms_field = 'sku'
__log_ltr_query_features(group[:1]["query_id"], key, doc_ids, click_prior_query, no_results, terms_field=terms_field)        
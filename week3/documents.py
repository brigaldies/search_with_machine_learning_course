#
# A simple endpoint that can receive documents from an external source, mark them up and return them.  This can be useful
# for hooking in callback functions during indexing to do smarter things like classification
#
from flask import (
    Blueprint, request, abort, current_app, jsonify
)
import fasttext
import json
import nltk
from nltk.stem import SnowballStemmer
nltk.download("punkt")
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")

bp = Blueprint('documents', __name__, url_prefix='/documents')

# Take in a JSON document and return a JSON document
@bp.route('/annotate', methods=['POST'])
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


# response[f'{item}_synonyms'] = ['list', 'of', 'synonyms']
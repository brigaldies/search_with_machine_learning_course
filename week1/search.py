#
# The main search hooks for the Search Flask application.
#
from flask import (
    Blueprint, redirect, render_template, request, url_for
)

from week1.opensearch import get_opensearch

import json

bp = Blueprint('search', __name__, url_prefix='/search')


# Process the filters requested by the user and return a tuple that is appropriate for use in: the query, URLs displaying the filter and the display of the applied filters
# filters -- convert the URL GET structure into an OpenSearch filter query
# display_filters -- return an array of filters that are applied that is appropriate for display
# applied_filters -- return a String that is appropriate for inclusion in a URL as part of a query string.  This is basically the same as the input query string
def process_filters(filters_input):
    # Filters look like: &filter.name=regularPrice&regularPrice.key={{ agg.key }}&regularPrice.from={{ agg.from }}&regularPrice.to={{ agg.to }}
    filters = []
    display_filters = []  # Also create the text we will use to display the filters that are applied
    applied_filters = ""
    for filter in filters_input:
        type = request.args.get(filter + ".type")
        display_name = request.args.get(filter + ".displayName", filter)
        print(f"Filter: {display_name}, type={type}")
        #
        # We need to capture and return what filters are already applied so they can be automatically added to any existing links we display in aggregations.jinja2
        applied_filters += "&filter.name={}&{}.type={}&{}.displayName={}".format(filter, filter, type, filter, display_name)
        
        #TODO: IMPLEMENT AND SET filters, display_filters and applied_filters.
        # filters get used in create_query below.  display_filters gets used by display_filters.jinja2 and applied_filters gets used by aggregations.jinja2 (and any other links that would execute a search.)
        if type == "range":
            price_range_from = request.args.get(filter + '.from')
            if not price_range_from:
                price_range_from = 0
            price_range_to = request.args.get(filter + '.to')
            if price_range_to:
                price_range_filter = {
                    "range": {
                        "regularPrice": {
                            "gte": price_range_from,
                            "lt": price_range_to
                        }
                    }
                }
            else:
                price_range_filter = {
                    "range": {
                        "regularPrice": {
                            "gte": price_range_from
                        }
                    }
                }
            filters.append(price_range_filter)
            applied_filters += f"&{filter}.from={price_range_from}&{filter}.to={price_range_to}"
            display_filters.append(filter)
        elif type == "terms":
            term_department_keyword = request.args.get(filter + ".key")
            filters.append({
                "term": {
                    "department.keyword": term_department_keyword
                }
            })
            applied_filters += f"&{filter}.key={term_department_keyword}"
            display_filters.append(filter)
    print(f"Filters          : {filters}")
    print(f"Applied filters  : {applied_filters}")
    print(f"Displayed filters: {display_filters}")
    return filters, display_filters, applied_filters


# Our main query route.  Accepts POST (via the Search box) and GETs via the clicks on aggregations/facets
@bp.route('/query', methods=['GET', 'POST'])
def query():
    opensearch = get_opensearch() # Load up our OpenSearch client from the opensearch.py file.
    # Put in your code to query opensearch.  Set error as appropriate.
    error = None
    user_query = None
    query_obj = None
    display_filters = None
    applied_filters = ""
    filters = None
    sort = "_score"
    sortDir = "desc"
    if request.method == 'POST':  # a query has been submitted
        user_query = request.form['query']
        if not user_query:
            user_query = "*"
        sort = request.form["sort"]
        if not sort:
            sort = "_score"
        sortDir = request.form["sortDir"]
        if not sortDir:
            sortDir = "desc"
        print(f"POST: user_query={user_query}, sort={sort}, sortDir={sortDir}")
        query_obj = create_query(user_query, [], sort, sortDir)
    elif request.method == 'GET':  # Handle the case where there is no query or just loading the page
        user_query = request.args.get("query", "*")
        filters_input = request.args.getlist("filter.name")
        sort = request.args.get("sort", sort)
        sortDir = request.args.get("sortDir", sortDir)
        if filters_input:
            (filters, display_filters, applied_filters) = process_filters(filters_input)
        print(f"GET: user_query={user_query}, sort={sort}, sortDir={sortDir}, filters={filters_input}")
        query_obj = create_query(user_query, filters, sort, sortDir)
    else:
        query_obj = create_query("*", [], sort, sortDir)

    # TODO: Replace me with an appropriate call to OpenSearch
    response = opensearch.search(
        body = query_obj,
        index = "bbuy_products"
    )

    print(f"Hits count: {response['hits']['total']['value']}")

    hits = response['hits']['hits']
    for i, hit in enumerate(hits):
        _source = hit['_source']
        print(f"[i+1] _id={hit['_id']}, score={hit['_score']}, name={_source['name']}, description={_source['shortDescription'] if 'shortDescription' in _source.keys() else ''}")

    debug_aggregations = False
    if debug_aggregations and 'aggregations' in response.keys():
        aggs = response['aggregations']
        print(f"Aggregations: {json.dumps(aggs, indent=4)}")

    # Postprocess results here if you so desire

    # Produce a (very!) crude Did-You-Mean suggestion.
    # This is a very crude term-level Did-You-Mean suggested, which was added without re-indexing.
    suggestions = []
    if 'suggest' in response.keys():
        # Select the best suggestion across those suggested for name, short, and long descriptions
        for k, v in response['suggest'].items():
            print(f"Suggestions for field {k}: {v}")
            field_suggestions = v
            field_suggestion = ''
            field_suggestion_score = 0
            for term_suggestion in v:
                options = term_suggestion['options']
                if len(options) > 0:
                    field_suggestion += ' ' + options[0]['text']
                    field_suggestion_score += options[0]['score']
                else:
                    field_suggestion += ' ' + term_suggestion['text']
            suggestions.append(
                {
                    "suggestion": field_suggestion.strip(),
                    "score": field_suggestion_score
                }
            )

    suggestions.sort(key=lambda x: x['score'], reverse=True)
    print(f"suggestions: {json.dumps(suggestions, indent = 4)}")
    
    #print(response)
    if error is None:
        return render_template("search_results.jinja2", query=user_query, search_response=response,
                               display_filters=display_filters, applied_filters=applied_filters,
                               sort=sort, sortDir=sortDir, suggest=suggestions[0] if (len(suggestions) > 0 and suggestions[0]['suggestion'] != user_query) else None)
    else:
        redirect(url_for("index"))


def create_query(user_query, filters, sort="_score", sortDir="desc"):
    print("Query: {} Filters: {} Sort: {}".format(user_query, filters, sort))
    if user_query == '' or user_query == '*':
        search_query = {
            "match_all": {}
        }
    else:
        search_query = {
            "multi_match": {
                "fields": [
                    "name^100",
                    "shortDescription^50",
                    "longDescription^10",
                    "department"
                ],
                "query": user_query
            }
        }
    
    highlighting = {
        "number_of_fragments" : 1,
        "fragment_size" : 100,
        "pre_tags" : ["<em>"],
        "post_tags" : ["</em>"],
        "fields" : {
            "name": {},
            "shortDescription": {},
            "longDescription": {}
        }
    }

    suggester_min_doc_frequency = 0.001
    suggester = {
        "text" : user_query,
        "name" : {
            "term" : {
                "field" : "name",
                # "analyzer": "standard",
                "suggest_mode": "missing",
                "min_doc_freq": suggester_min_doc_frequency
            }
        },
        "shortDescription" : {
            "term" : {
                "field" : "shortDescription",
                # "analyzer": "standard",
                "suggest_mode": "missing",
                "min_doc_freq": suggester_min_doc_frequency
            }
        },
        "longDescription" : {
            "term" : {
                "field" : "longDescription",
                # "analyzer": "standard",
                "suggest_mode": "missing",
                "min_doc_freq": suggester_min_doc_frequency
            }
        }
    }

    query_obj = {
        'size': 10,
        # Build a query that both searches and filters
        "query": {
            "bool": {
                "must": [
                    {
                        "function_score": {
                            "query": search_query,
                            "boost_mode": "multiply",
                            "score_mode": "avg",
                            "functions": [
                                {
                                "field_value_factor": {
                                    "field": "salesRankShortTerm",
                                    "factor": 1,
                                    "modifier": "reciprocal",
                                    "missing": 100000000
                                }
                                },
                                {
                                "field_value_factor": {
                                    "field": "salesRankMediumTerm",
                                    "factor": 1,
                                    "modifier": "reciprocal",
                                    "missing": 100000000
                                }
                                },
                                {
                                "field_value_factor": {
                                    "field": "salesRankLongTerm",
                                    "factor": 1,
                                    "modifier": "reciprocal",
                                    "missing": 100000000
                                }
                                }
                            ]
                        }
                    } 
                ],
                "filter": filters
            }
        },
        "sort": [
            {
                sort: {
                    "order": sortDir
                }
            }
        ],
        "highlight": highlighting,
        "suggest": suggester,
        "aggs": {
            "department": {
                "terms": {
                    "field": "department.keyword"
                }
            },
            "regularPrice": {
                "range": {
                    "field": "regularPrice",
                    "ranges": [
                        { "key": "$", "to": 10 },
                        { "key": "$$", "from": 10, "to": 50 },
                        { "key": "$$$", "from": 50, "to": 100 },
                        { "key": "$$$$", "from": 100, "to": 500 },
                        { "key": "$$$$$", "from": 500, "to": 1000 },
                        { "key": "$$$$$$", "from": 1000 }
                    ]
                }
            },
            "missing_images": {
                "missing": {
                    "field": "image"
                }
            }
        }
    }
    print("query obj: {}".format(query_obj))
    return query_obj

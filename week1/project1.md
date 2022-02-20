# Week 1 Project

## Level 1

See Self-Assessment section further down.

## Level 2: Bad relevancy for q=Apple IPad 2

Why is the ranking is bad for q=apple iPad 2?

1. The multi-field query does not reward phrase matches.
    - Remedy: Add a boosted "phrase_match" clause wrapped in a Bool/Should query, with slop=2 due to the copyright symbol (R) being tokenized as an individual term.
1. Repetitions of terms (higher TF), which reward the score, do not represent relevancy in e-commercer (short text).
    - Remedy: Saturate TF to a constant 1.
1. Matching on both shortDescription and longDescription increases the score even though the relevancy "signal" is the same. For example, _id=2339322 does not match on shortDescription even though it is the expected top result, while the top result _id=3640066 matches on both fields.
    - Remedy: Index both the short and long description in the same search field.

### Score Analysis

#### Expected best result: "Apple® - iPad® 2 with Wi-Fi - 16GB - White”, _id=2339322

```
# Explain score
POST /bbuy_products/_explain/2339322
{
  "query": {
    "multi_match": {
      "fields": [
        "name^100",
        "shortDescription^50",
        "longDescription^10",
        "department"
      ],
      "query": "apple ipad 2"
    }
  }
}
```
```
Max of:
Sum (
    longDescription:ipad —> 67.04193
    longDescription:2 —> 30.95195
) = 97.99388
|
Sum (
    name:appl —> 414.17224
    name:pad —> 526.4822
    name:2 —> 234.97534
) = 1175.6298
= 175.6298
``` 
```json
{
            "value" : 526.4822,
            "description" : "weight(name:ipad in 103038) [PerFieldSimilarity], result of:",
            "details" : [
              {
                "value" : 526.4822,
                "description" : "score(freq=1.0), computed as boost * idf * tf from:",
                "details" : [
                  {
                    "value" : 220.0,
                    "description" : "boost",
                    "details" : [ ]
                  },
                  {
                    "value" : 6.850972,
                    "description" : "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:",
                    "details" : [
                      {
                        "value" : 1359,
                        "description" : "n, number of documents containing term",
                        "details" : [ ]
                      },
                      {
                        "value" : 1284453,
                        "description" : "N, total number of documents with field",
                        "details" : [ ]
                      }
                    ]
                  },
                  {
                    "value" : 0.3493082,
                    "description" : "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
                    "details" : [
                      {
                        "value" : 1.0,
                        "description" : "freq, occurrences of term within document",
                        "details" : [ ]
                      },
                      {
                        "value" : 1.2,
                        "description" : "k1, term saturation parameter",
                        "details" : [ ]
                      },
                      {
                        "value" : 0.75,
                        "description" : "b, length normalization parameter",
                        "details" : [ ]
                      },
                      {
                        "value" : 9.0,
                        "description" : "dl, length of field",
                        "details" : [ ]
                      },
                      {
                        "value" : 5.1830015,
                        "description" : "avgdl, average length of field",
                        "details" : [ ]
                      }
                    ]
                  }
                ]
              }
            ]
          }
```

#### Top non-relevant match: "Rocketfish™ - Bluetooth Speaker for Apple® iPad®, iPad 2 and iPad (3rd Generation)", _id=3640066

```
# Explain score
POST /bbuy_products/_explain/3640066
{
  "query": {
    "multi_match": {
      "fields": [
        "name^100",
        "shortDescription^50",
        "longDescription^10",
        "department"
      ],
      "query": "apple ipad 2"
    }
  }
}
```
```
Other smaller-value clause scores are not shown.
Sum (
    name:appl —> 333.30518
    Name:pad —> 813.6284
    Name:2 —> 189.096444
) = 1336.03
```
```json
{
            "value" : 813.6284,
            "description" : "weight(name:ipad in 64216) [PerFieldSimilarity], result of:",
            "details" : [
              {
                "value" : 813.6284,
                "description" : "score(freq=3.0), computed as boost * idf * tf from:",
                "details" : [
                  {
                    "value" : 220.0,
                    "description" : "boost",
                    "details" : [ ]
                  },
                  {
                    "value" : 6.850972,
                    "description" : "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:",
                    "details" : [
                      {
                        "value" : 1359,
                        "description" : "n, number of documents containing term",
                        "details" : [ ]
                      },
                      {
                        "value" : 1284453,
                        "description" : "N, total number of documents with field",
                        "details" : [ ]
                      }
                    ]
                  },
                  {
                    "value" : 0.5398228,
                    "description" : "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
                    "details" : [
                      {
                        "value" : 3.0,
                        "description" : "freq, occurrences of term within document",
                        "details" : [ ]
                      },
                      {
                        "value" : 1.2,
                        "description" : "k1, term saturation parameter",
                        "details" : [ ]
                      },
                      {
                        "value" : 0.75,
                        "description" : "b, length normalization parameter",
                        "details" : [ ]
                      },
                      {
                        "value" : 13.0,
                        "description" : "dl, length of field",
                        "details" : [ ]
                      },
                      {
                        "value" : 5.1830015,
                        "description" : "avgdl, average length of field",
                        "details" : [ ]
                      }
                    ]
                  }
                ]
              }
            ]
          }
```

Larger score (813.6284 vs 526.4822) for “iPad” because of the TF=3 
Is TF applicable in e-commerce????

# Self-Assessment Questions :male-detective:
## Do your counts match ours?
### Number of documents in the Product index: 1,275,077
Yes; See query below.
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
```
### Number of documents in the Query Log index: 1,865,269
Yes; See query below.
```
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
### There are 16,772 items in the “Computers” department when using a “match all” query (“*”) and faceting on “department.keyword”.
Yes; See query below.
```
POST /bbuy_products/_search
{
  "size": 0,
  "query": {
    "bool": {
      "must": [
        {
          "match_all": {}
        }
      ],
      "filter": [
        {
          "terms": {
            "department.keyword": ["COMPUTERS"]
          }
        }
      ]
    }
  },
  "aggs": {
    "department": {
      "terms": {
        "field": "department.keyword"
      }
    }
  }
}

Result:
{
  "took" : 11,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 10000,
      "relation" : "gte"
    },
    "max_score" : null,
    "hits" : [ ]
  },
  "aggregations" : {
    "department" : {
      "doc_count_error_upper_bound" : 0,
      "sum_other_doc_count" : 0,
      "buckets" : [
        {
          "key" : "COMPUTERS",
          "doc_count" : 16772
        }
      ]
    }
  }
}
```
### Number of documents missing an “image” field: 4,434
Yes; See query below.
```
POST /bbuy_products/_search
{
  "size": 0,
  "query": {
    "bool": {
      "must": [
        {
          "match_all": {}
        }
      ],
      "filter": [
      ]
    }
  },
  "aggs": {
    "missing_images": {
      "missing": {
        "field": "image"
      }
    }
  }
}

Result:
{
  "took" : 41,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 10000,
      "relation" : "gte"
    },
    "max_score" : null,
    "hits" : [ ]
  },
  "aggregations" : {
    "missing_images" : {
      "doc_count" : 4434
    }
  }
}
```
## What field types and analyzers did you use for the following fields and why?
### Name, shortDescription, and longDescription
```
"name": {
                "type": "text",
                "analyzer": "english"
            },
```
For ```name```, ```shortDescription```, and ```longDescription```, a field type "text" with the built-in "english" analyzer is used in order to support free text-based search, including text analysis behavior such as stop words removal and stemming.

### regularPrice
```
"regularPrice": {
                "type": "float"
            },
```
For ```regularPrice``` a field type "float is used to support price range queries, filters, and facets.

## Compare your Field mappings with the instructors. Where did you align and where did you differ? What are the pros and cons to the different approaches?
I believe I am aligned with the instructions (see criteria below).

See my OpenSearch mappings in my repo ```./opensearch/index-bbuy-products.mappings.json``` amd ```./opensearch/index-bbuy-queries.mappings.json```

### You should have one field mapping entry for every field specified in the index-bbuy.logstash XPath filter
Done.
### All fields that are type “text” should also have a “keyword” multi-field
Done.
### The “regularPrice” field (or a multi-field variant of it) must support doing numeric range queries and numeric range aggregations
Done.

## Were you able to get the “ipad 2” to show up in the top of your results? How many iterations did it take for you to get there, if at all? (we’re not scoring this, of course, it’s just worth noting that hand tuning like this can often be quite time consuming.)

I followed the "relevancy tuning journey" as prescribed in the project's instructions.

Starting with the base query, which produces irrelevant results at the top (see my score analysis at the top of this report):
```
GET bbuy_products/_search
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "query_string": {
            "query": "\"ipad 2\"",
            "fields": [
              "name^100",
              "shortDescription^50",
              "longDescription^10",
              "department"
            ]
          }
        }
      ]
    }
  }
}
```

Then, the "boosting" query with a "negative" component to de-boost observed irrelevant matches, which despite the many terms and phrases being de-boosted still does not produce the expected top result (_id=2339322 for name="Apple® - iPad® 2 with Wi-Fi - 16GB - White")
```
GET bbuy_products/_search
{
  "size": 10,
  "query": {
    "boosting": {
      "positive": {
        "query_string": {
          "query": "\"ipad 2\"",
          "fields": [
            "name^100",
            "shortDescription^50",
            "longDescription^10",
            "department"
          ]
        }
      },
      "negative": {
        "bool": {
          "should": [
            {
              "query_string": {
                "query": "\"compatible iPad\"~20 OR \"fit iPad\"~3 OR \"outfit iPad\"~3 OR \"connect iPad\"~3 OR protector OR protect OR jacket OR headset OR microphone OR keyboard OR charging OR adapter OR speaker OR stylus OR sleeve OR \"hands free\"",
                "fields": [
                  "name",
                  "shortDescription",
                  "longDescription"
                ]
              }
            }
          ]
        }
      },
      "negative_boost": 0.01
    }
  }
}
```

And finally the "function score" query with either the "multiply" or "replace" boost mode, which does produce the expected result at the top:
```
GET bbuy_products/_search
{
  "size": 10,
  "query": {
    "function_score": {
      "query": {
        "query_string": {
          "query": "\"ipad 2\"",
          "fields": [
            "name^1000",
            "shortDescription^50",
            "longDescription^10",
            "department"
          ]
        }
      },
      "boost_mode": "replace",
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
}
```
Hand-tuning relevancy is a "whack-a-mole" game, which is very frustrating, hence this class. :-)

## Highlighting

- Added the "highlight" section in the OpenSearch query
```
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
```
- Added the css style to make the highlighted text bold and italic:
```
em {
    font-style: italic;
    font-weight: bold;
}
```

# Level 3

## Pagination

Not started.

## Kibana Dashboards

Not started.

## Spell Checking

A very crude spell checking/Did-you-mean functionality was added without requiring re-indexing by adding the following "suggest" component in the OpenSearch search request:
```
suggester_min_doc_frequency = 0.001
    suggester = {
        "text" : user_query,
        "name" : {
            "term" : {
                "field" : "name",
                "analyzer": "standard",
                "suggest_mode": "missing",
                "min_doc_freq": suggester_min_doc_frequency
            }
        },
        "shortDescription" : {
            "term" : {
                "field" : "shortDescription",
                "analyzer": "standard",
                "suggest_mode": "missing",
                "min_doc_freq": suggester_min_doc_frequency
            }
        },
        "longDescription" : {
            "term" : {
                "field" : "longDescription",
                "analyzer": "standard",
                "suggest_mode": "missing",
                "min_doc_freq": suggester_min_doc_frequency
            }
        }
    }
```
### Further Spell-Checking Work
1. Re-index an all-text (from name, short and long descriptions) field without stemming. Spell check dictionaries are typically built from un-stemmed terms.
1. Tune the [many suggest parameters](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-suggesters.html) such as suggest mode, min doc frequency, etc.
1. Build a spell check dictionary from the queries log.

## Auto-Suggestion

Not started.

## Query Re-Writing

Not started.

## Synonyms Expansion

A very crude query-time synonyms expansion mechanism was added with the following index settings update (Note: no re-indexing was necessary since the mechanism uses query-time synonyms expansion and not index-time):

### View current settings
```
GET /bbuy_products/_settings
```

### Close index
The index must be closed in order to update its settings.
```
POST /bbuy_products/_close
```
### Create a Synonyms Filter and Analyzer
```json
PUT bbuy_products/_settings
{
  "analysis": {
    "filter": {
      "synonyms_filter": {
        "type": "synonym",
        "synonyms": [
          "computer,laptop,desktop"
        ]
      }
    },
    "analyzer": {
      "synonyms_analyzer": {
        "tokenizer": "standard",
        "filter": [
          "lowercase",
          "synonyms_filter"
        ]
      }
    }
  }
}
```

### Re-Open index
```
POST /bbuy_products/_open
```

### Test Query-Time Synonyms Expansion
```json
GET /bbuy_products/_validate/query?explain
{
  "query": {
      "query_string": {
         "fields": ["name^100", "shortDescription^50", "longDescription"],
         "query": "apple computer",
         "analyzer": "synonyms_analyzer"
      }
  }
}

Response:
{
      "index" : "bbuy_products",
      "valid" : true,
      "explanation" : "((longDescription:apple Synonym(longDescription:computer longDescription:desktop longDescription:laptop)) | (name:apple Synonym(name:computer name:desktop name:laptop))^100.0 | (shortDescription:apple Synonym(shortDescription:computer shortDescription:desktop shortDescription:laptop))^50.0)"
    }
```

### Search With Query-Time Synonyms Expansion
```json
# Search with query-time synonyms expansion
POST /bbuy_products/_search
{
  "query": {
    "multi_match": {
      "fields": [
        "name^100",
        "shortDescription^50",
        "longDescription^10",
        "department"
      ],
      "query": "apple computer",
      "analyzer":   "synonyms_analyzer"
    }
  }
}
```

### Further Synonyms Work

1. 

## Analyze and Index Height/Width Unit of Measure
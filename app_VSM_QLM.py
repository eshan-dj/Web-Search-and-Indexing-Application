import json
import math
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from collections import defaultdict

# final inverted index
with open(r"D:\NIBM\HNDS\Information retrival\IR_Project_2\final_inverted_index.json") as f:
    inverted_index = json.load(f)

# Get total number of documents
all_doc_ids = set()
for doc_list in inverted_index.values():
    all_doc_ids.update(doc_list)
N = len(all_doc_ids)  

# Function to compute TF-IDF ranking 
def rank_documents_vsm(query):
    query_terms = query.lower().split()
    doc_scores = defaultdict(float)
    
    for term in query_terms:
        if term in inverted_index:
            doc_list = inverted_index[term]
            df = len(doc_list)  
            idf = math.log((N + 1) / (df + 1))  # IDF with smoothing
            
            for doc in doc_list:
                tf = 1 + math.log(doc_list.count(doc))  
                doc_scores[doc] += tf * idf  # TF-IDF score
    
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc, "score": score} for doc, score in ranked_docs]

#  Query Likelihood Model (QLM) with Dirichlet smoothing
def rank_documents_qlm(query):
    query_terms = query.lower().split()
    doc_scores = defaultdict(float)
    mu = 1000  
    
    for term in query_terms:
        if term in inverted_index:
            doc_list = inverted_index[term]
            term_count = sum(len(doc_list) for doc in inverted_index.values())  
            
            for doc in doc_list:
                tf = doc_list.count(doc)  
                doc_len = sum(len(inverted_index[t]) for t in inverted_index if doc in inverted_index[t])  
                prob = (tf + mu * (len(doc_list) / term_count)) / (doc_len + mu)
                doc_scores[doc] += math.log(prob) 
    
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc, "score": score} for doc, score in ranked_docs]


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    html.H1("VSM & QLM"),
    dcc.Input(id="query-input", type="text", placeholder="Enter your search query", debounce=True, style={"width": "80%"}),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label="VSM Results", children=[html.Ul(id="vsm-results")]),
        dcc.Tab(label="QLM Results", children=[html.Ul(id="qlm-results")])
    ])
])


@app.callback(
    Output("vsm-results", "children"),
    Output("qlm-results", "children"),
    Input("query-input", "value")
)
def update_results(query):
    if not query:
        return [], []
    
    vsm_results = rank_documents_vsm(query)
    qlm_results = rank_documents_qlm(query)

    vsm_list = [html.Li(f"{doc['doc_id']} - Score: {doc['score']:.2f}") for doc in vsm_results]
    qlm_list = [html.Li(f"{doc['doc_id']} - Score: {doc['score']:.2f}") for doc in qlm_results]

    return vsm_list, qlm_list

if __name__ == "__main__":
    app.run(debug=True)



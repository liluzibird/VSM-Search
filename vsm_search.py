#-------------------------------------------------------------------------
# AUTHOR: Albert Huynh
# FILENAME: vsm_search
# SPECIFICATION: Search engine using VSM to load information, converts to TF-IDF vectors, rank docs, sort queries based on AP, calculates AP and MAP
# FOR: CS 4250- Assignment #4
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# --> add your Python code here
documents = pd.read_csv("docs.csv")
queries = pd.read_csv("queries.csv")
relevance_judgments = pd.read_csv("relevance_judgments.csv")


# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# --> add your Python code here
vectorizer = TfidfVectorizer(stop_words="english")
document_matrix = vectorizer.fit_transform(documents["text"])


# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# --> add your Python code here
ap_scores = []

for query_index in range(len(queries)):
    query_id = queries.iloc[query_index]["query_id"]
    query_text = queries.iloc[query_index]["query_text"]

    query_vector = vectorizer.transform([query_text])
    similarity_scores = cosine_similarity(query_vector, document_matrix).flatten()
    ranked_document_indexes = similarity_scores.argsort()[::-1]

    query_judgments = relevance_judgments[
        relevance_judgments["query_id"] == query_id
    ]
    relevance_by_document = dict(
        zip(query_judgments["doc_id"], query_judgments["judgment"])
    )
    total_relevant = (query_judgments["judgment"] == "R").sum()

    relevant_found = 0
    precision_sum = 0


    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------

    # store the AP value for this query (use any data structure you prefer)
    for rank in range(len(ranked_document_indexes)):
        document_index = ranked_document_indexes[rank]
        document_id = documents.iloc[document_index]["doc_id"]

        if relevance_by_document[document_id] == "R":
            relevant_found += 1
            precision = relevant_found / (rank + 1)
            precision_sum += precision

    if total_relevant > 0:
        average_precision = precision_sum / total_relevant
    else:
        average_precision = 0

    ap_scores.append((query_id, average_precision))


# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
# --> add your Python code here
ap_scores.sort(key=lambda item: item[1], reverse=True)
map_score = sum(item[1] for item in ap_scores) / len(ap_scores)


# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("================================================================")
print("Queries sorted by Average Precision (AP):")
# --> add your Python code here
for query_id, average_precision in ap_scores:
    print(f"{query_id}: AP = {average_precision:.3f}")

print("================================================================")
print(f"MAP = {map_score:.3f}")

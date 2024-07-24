#Name:Abdi Dawud, section-3,ID:-UGR/25390/14

import os
import json
import math
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Ensure the Exsistances of   necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initializing stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Initializing the inverted index and document metadata
inverted_index = defaultdict(list)
document_metadata = {}

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def add_to_index(doc_id, terms):
    term_freq = defaultdict(int)
    for term in terms:
        term_freq[term] += 1
    for term, freq in term_freq.items():
        inverted_index[term].append((doc_id, freq))

def indexdoc(doc_dir):
    doc_id = 1
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(doc_dir, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                document_metadata[doc_id] = content
                terms = preprocess_text(content)
                add_to_index(doc_id, terms)
                doc_id += 1

def save_index_to_file(inverted_index, metadata, index_file, metadata_file):
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)

def load_index_from_file(index_file, metadata_file):
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            document_metadata = json.load(f)
        return inverted_index, document_metadata
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

def compute_tfidf(term, doc_id, term_freq, doc_freq, N):
    tf = term_freq[term]
    idf = math.log(N / (1 + doc_freq[term]))
    return tf * idf

def get_snippet(text, query, window=50):
    query_terms = query.lower().split()
    start = min(text.lower().find(term) for term in query_terms if term in text.lower())
    start = max(0, start - window // 2)
    end = min(len(text), start + window)
    snippet = text[start:end]
    return snippet

def search_query(query, inverted_index, document_metadata, doc_freq, N):
    query_terms = preprocess_text(query)
    scores = {}
    for term in query_terms:
        if term in inverted_index:
            for doc_id, freq in inverted_index[term]:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += compute_tfidf(term, doc_id, Counter(query_terms), doc_freq, N)
    
    # Rank documents by scores
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results = []
    for doc_id, score in ranked_docs:
        snippet = get_snippet(document_metadata[str(doc_id)], query)
        results.append((doc_id, snippet))
    return results

# Directory containing the text files
doc_dir = 'C:\\Users\\PC\\Desktop\\ML\\python\\project_SEO\\txt_files'
indexdoc(doc_dir)

# File paths
index_file = 'inverted_index.json'
metadata_file = 'document_metadata.json'
save_index_to_file(inverted_index, document_metadata, index_file, metadata_file)

# Load the data
loaded_inverted_index, loaded_document_metadata = load_index_from_file(index_file, metadata_file)

# Example search
N = len(loaded_document_metadata)
doc_freq = {term: len(postings) for term, postings in loaded_inverted_index.items()}
query = "example search query"
results = search_query(query, loaded_inverted_index, loaded_document_metadata, doc_freq, N)
for doc_id, content in results:
    print(f"Document ID: {doc_id}, Snippet: {content}")

# Retrieval Effectiveness Evaluation Functions
def precision(relevant_docs, retrieved_docs):
    return len(set(relevant_docs) & set(retrieved_docs)) / len(retrieved_docs) if retrieved_docs else 0.0

def recall(relevant_docs, retrieved_docs):
    return len(set(relevant_docs) & set(retrieved_docs)) / len(relevant_docs) if relevant_docs else 0.0

def f1_score(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def mean_average_precision(relevant_docs_list, retrieved_docs_list):
    total_precision = 0.0
    num_queries = len(relevant_docs_list)
    for relevant_docs, retrieved_docs in zip(relevant_docs_list, retrieved_docs_list):
        precisions = [precision(relevant_docs, retrieved_docs[:k+1]) for k in range(len(retrieved_docs))]
        if relevant_docs:
            total_precision += sum(precisions) / len(relevant_docs)
    return total_precision / num_queries if num_queries else 0.0

# Example usage with dummy data
relevant_docs_list = [[1, 2, 3], [2, 3, 4]]
retrieved_docs_list = [[1, 2, 4], [2, 3, 5]]
prec = precision(relevant_docs_list[0], retrieved_docs_list[0])
rec = recall(relevant_docs_list[0], retrieved_docs_list[0])
f1 = f1_score(prec, rec)
map_score = mean_average_precision(relevant_docs_list, retrieved_docs_list)
print(f"Precision: {prec}, Recall: {rec}, F1-Score: {f1}, MAP: {map_score}")

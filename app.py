#Name:Abdi Dawud, section-3,ID:-UGR/25390/14
from flask import Flask, request, render_template
from index import search_query, load_index_from_file

app = Flask(__name__)

# Global variables for the loaded index and metadata
loaded_inverted_index = None
loaded_document_metadata = None
doc_freq = None
N = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    global loaded_inverted_index, loaded_document_metadata, doc_freq, N
    query = request.form['query']
    results = search_query(query, loaded_inverted_index, loaded_document_metadata, doc_freq, N)
    if not results:
        return render_template('results.html', query=query, results=[("No documents found.", "")])
    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    # Load the index and document metadata from files
    index_file = 'inverted_index.json'
    metadata_file = 'document_metadata.json'
    
    # Debugging prints
    print(f"Loading index from {index_file} and metadata from {metadata_file}")
    
    loaded_inverted_index, loaded_document_metadata = load_index_from_file(index_file, metadata_file)
    
    print(f"Loaded inverted index: {loaded_inverted_index}")
    print(f"Loaded document metadata: {loaded_document_metadata}")
    
    # Ensure that the loaded data is not None
    if loaded_inverted_index is None or loaded_document_metadata is None:
        raise ValueError("Failed to load index or metadata. Check the files and their contents.")
    
    # Calculate the total number of documents
    N = len(loaded_document_metadata)
    print(f"Total number of documents: {N}")
    
    # Calculate document frequency for each term
    doc_freq = {term: len(postings) for term, postings in loaded_inverted_index.items()}
    print(f"Document frequency: {doc_freq}")
    
    # Run the Flask app
    app.run(debug=True, port=5000)
    #use  http://127.0.0.1:5000 for web brosower
     
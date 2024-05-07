import warnings

# If you can import the warning class directly
from langchain.document_loaders import LangChainDeprecationWarning

# Ignore LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from datasets import load_dataset
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader

load_dotenv()  
from Bio import Entrez
import csv
import os
import shutil
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document

Entrez.email = 'jessicabull833@example.com'

CHROMA_PATH = "chroma"

def search_pubmed(query, retmax=10, year=None):
    """Search PubMed for the given query and year, return the list of PubMed IDs."""
    # If a year is specified, refine the query to include it in the search
    if year:
        query = f"{query} AND {year}[Date - Publication]"

    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, retmode="xml")
    results = Entrez.read(handle)
    return results['IdList']


# def load_documents(pub_ids, output_file='documents.csv', retmax=10):
#     """Fetches documents from PubMed and saves them to a CSV file."""
#     documents = []  # Initialize the documents list\
#     metadata = []
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['pub_id', 'title', 'abstract', 'authors', 'year', 'url']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
#         writer.writeheader()

#         for i in range(0, len(pub_ids), retmax):
#             j = i + retmax
#             if j > len(pub_ids):
#                 j = len(pub_ids)
#             handle = Entrez.efetch(db="pubmed", id=','.join(pub_ids[i:j]),
#                                    rettype="xml", retmode="text", retmax=retmax)
#             records = Entrez.read(handle)

#             for pubmed_article in records['PubmedArticle']:
#                 try:
#                     pmid = str(pubmed_article['MedlineCitation']['PMID'])
#                     title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
#                     abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0] \
#                         if 'Abstract' in pubmed_article['MedlineCitation']['Article'] else "No abstract available"
#                     authors_list = pubmed_article['MedlineCitation']['Article'].get('AuthorList', [])
#                     authors = '; '.join([author.get('LastName', '') + ", " + author.get('ForeName', '') for author in authors_list])
#                     year = pubmed_article['MedlineCitation']['Article'].get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', 'Unknown')
#                     pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#                     metadata.append({
#                         'pub_id': pmid,
#                         'title': title,
#                         'abstract': abstract,
#                         'authors': authors,
#                         'year': year,
#                         'url': pubmed_url
#                     })
#                     writer.writerow({
#                         'pub_id': pmid,
#                         'title': title,
#                         'abstract': abstract,
#                         'authors': authors,
#                         'year': year,
#                         'url': pubmed_url
#                     })

#                     # Also append to the documents list for loading into Chroma
#                     documents.append({
#                         'pub_id': pmid,
#                         'title': title,
#                         'abstract': abstract,
#                         'authors': authors,
#                         'year': year,
#                         'url': pubmed_url
#                     })
#                 except KeyError:
#                     continue

#     return documents
# # print(documents)

# def save_to_chroma(documents):
#     """Saves documents to Chroma DB."""

#     load_dotenv()

#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if openai_api_key is None:
#         raise ValueError("OPENAI_API_KEY is not set. Please ensure your .env file contains this variable.")

#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key))

#     try:
#         with open(os.path.join(CHROMA_PATH, 'existing_ids3.txt'), 'r') as file:
#             existing_pub_ids = set(file.read().splitlines())
#     except FileNotFoundError:
#         existing_pub_ids = set()

#     documents_for_chroma = []
#     new_documents_count = 0
#     new_ids = set()

#     for doc in documents:
#         if doc['pub_id'] not in existing_pub_ids:
#             document_for_chroma = Document(
#                 page_content=f"{doc['title']} {doc['abstract']}",
#                 metadata=doc
#             )
#             documents_for_chroma.append(document_for_chroma)
#             new_documents_count += 1
#             new_ids.add(doc['pub_id'])

#     if documents_for_chroma:
#         db.add_documents(documents_for_chroma)
#         db.persist()
#         with open(os.path.join(CHROMA_PATH, 'existing_ids3.txt'), 'a') as file:
#             for pub_id in new_ids:
#                 file.write(f"{pub_id}\n")
#         print(f"Added {new_documents_count} new documents to {CHROMA_PATH}.")
#     else:
#         print("No new documents to add.")

import os
import csv
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from Bio import Entrez

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pc = Pinecone(api_key=api_key)

# Initialize OpenAIEmbeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set. Please ensure your .env file contains this variable.")
embed_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Function to split text into smaller chunks
def split_text(text, chunk_size=10):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
def split_text(text):
    return list(text)

# Function to save data to Pinecone
def save_to_pinecone(documents, index_name):
    index = pc.Index(index_name)

    for doc in documents:
        # doc_id = f"{doc['pub_id']}"
        # if index.fetch(ids=[doc_id]).get('results') == {}:
            # Split title and abstract into smaller chunks
            title_chunks = doc['title']
            # Embed each chunk separately
            for title_chunk in title_chunks:
                # for abstract_chunk in abstract_chunks:
                text_chunk = f"{title_chunk}"
                embedding = embed_model.embed_documents(text_chunk)
                # Flatten the embedding list
                flattened_embedding = [item for sublist in embedding for item in sublist]
                # Convert flattened embedding to floats
                embedding_float = [float(value) for value in flattened_embedding]
                # Generate unique ID for the document
                doc_id = f"{doc['pub_id']}"
                # Store metadata
                metadata = {
                    'pub_id': doc['pub_id'],
                    'title': doc['title'],
                    'abstract': doc['abstract'],
                    'authors': doc['authors'],
                    'year': doc['year'],
                    'url': doc['url']
                }
                # Upsert into Pinecone index
                index.upsert(vectors=[(doc_id, embedding_float, metadata)])
            # else:
            #     print("already exists")

# Function to search PubMed
def search_pubmed(query, retmax=100, year=None):
    if year:
        query = f"{query} AND {year}[Date - Publication]"

    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, retmode="xml")
    results = Entrez.read(handle)
    return results['IdList']

# Function to load documents from PubMed
def load_documents(pub_ids, retmax=100):
    documents = []
    for i in range(0, len(pub_ids), retmax):
        j = min(i + retmax, len(pub_ids))
        handle = Entrez.efetch(db="pubmed", id=pub_ids[i:j], rettype="xml", retmode="text", retmax=retmax)
        records = Entrez.read(handle)

        for pubmed_article in records['PubmedArticle']:
            try:
                pmid = str(pubmed_article['MedlineCitation']['PMID'])
                title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
                abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0] \
                    if 'Abstract' in pubmed_article['MedlineCitation']['Article'] else "No abstract available"
                authors_list = pubmed_article['MedlineCitation']['Article'].get('AuthorList', [])
                authors = '; '.join([author.get('LastName', '') + ", " + author.get('ForeName', '') for author in authors_list])
                year = pubmed_article['MedlineCitation']['Article'].get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', 'Unknown')
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                documents.append({
                    'pub_id': pmid,
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'year': year,
                    'url': pubmed_url
                })
            except KeyError:
                continue
    return documents

def main():
    query = "biology"
    index_name = "pinecone"
    for year in range(2024, 1999, -1):
        print(f"Processing year: {year}")
        pub_ids = search_pubmed(query, retmax=2500, year=str(year))
        if pub_ids:
            documents = load_documents(pub_ids)
            save_to_pinecone(documents, index_name)
        else:
            print(f"No publications found for {year}")

if __name__ == '__main__':
    main()
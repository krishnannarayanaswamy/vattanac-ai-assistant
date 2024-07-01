from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from pathlib import Path
from constants import ASTRA_COLLECTION_NAME
import json
from langchain_experimental.text_splitter import SemanticChunker

import os

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
keyspace=os.environ['ASTRA_DB_KEYSPACE']
openai_api_key=os.environ["OPENAI_API_KEY"]

def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

vstore = AstraDBVectorStore(
    embedding=get_embeddings_model(),
    collection_name=ASTRA_COLLECTION_NAME,
    api_endpoint=api_endpoint,
    token=token,
    namespace=keyspace,
)

#text_splitter = SemanticChunker(
#    get_embeddings_model(), breakpoint_threshold_type="percentile"
#)
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 4096,
    chunk_overlap  = 50,
    length_function = len,
        separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\n---\n"
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)


with open("files/annual_report_2021_khmer.txt") as f:
    annual_report = f.read()

docs = text_splitter.create_documents([annual_report])
#print(docs[0].page_content)

print(len(docs))

inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")

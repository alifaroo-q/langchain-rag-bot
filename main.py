import os
from dotenv import load_dotenv

from langchain_voyageai import VoyageAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_postgres import PGVector

load_dotenv()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
voyage_api_key = os.getenv('VOYAGE_API_KEY')

embeddings = VoyageAIEmbeddings(
    voyage_api_key=voyage_api_key, model="voyage-2"
)

loader = TextLoader('facts.txt')

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0

)

docs = loader.load_and_split(text_splitter=text_splitter)

db = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name='docs',
    connection=connection,
    use_jsonb=True
)


result = db.similarity_search(
    'what is an interesting fact about the English language?'
)

for res in result:
    print("\n")
    print(res)

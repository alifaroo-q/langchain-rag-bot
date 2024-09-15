import os
from dotenv import load_dotenv

from langchain_anthropic import Anthropic
from langchain_voyageai import VoyageAIEmbeddings

from langchain_postgres import PGVector

from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

voyage_api_key = os.getenv('VOYAGE_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

chat_model = Anthropic(anthropic_api_key=anthropic_api_key)

embeddings = VoyageAIEmbeddings(
    voyage_api_key=voyage_api_key, model="voyage-2"
)

db = PGVector(
    embeddings=embeddings,
    connection=connection,
    collection_name='docs',
    use_jsonb=True
)

retriever = db.as_retriever()

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    chat_model,
    retrieval_qa_chat_prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)

result = rag_chain.invoke(
    {"input": "What is an interesting fact about English language?"})

print(result)

print(result['answer'])

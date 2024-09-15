# Retrieval Augmented Generation (RAG) | Chat bot

## PGVector

use the following docker command to spin up pgvector container

```bash
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
```

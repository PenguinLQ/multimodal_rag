from pymilvus import MilvusClient, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict
from multimodal_rag.common.config import get_settings

settings = get_settings()

"""Store text and image content and embeddings into Milvus"""
milvus_client = MilvusClient(uri=settings.data_dir + "/" + settings.db_file)
embedding_dim = settings.dimension


def create_collections(text_collection_name:str, image_collection_name: str):
    # create text collection fields
    text_col_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="article_title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]

    # create image collection fields
    image_col_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="article_title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR,  max_length=512),
        FieldSchema(name="caption", dtype=DataType.VARCHAR,  max_length=512),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]

    # create text collection
    create_milvus_collection(text_collection_name, "text collection", text_col_fields)
    # create image collection
    create_milvus_collection(image_collection_name, "image collection", image_col_fields)

def create_milvus_collection(name: str, description: str, fields: List[FieldSchema]):
    # Drop the collection if it already exists
    if milvus_client.has_collection(name):
        milvus_client.drop_collection(name)

    # create collection schema
    schema = CollectionSchema(fields=fields, description=description)

    milvus_client.create_collection(
        collection_name=name,
        schema=schema
    )

def insert_data(collection_name: str, data: List[Dict]) -> Dict:
    insert_result = milvus_client.insert(collection_name=collection_name, data=data)
    print("Inserted", insert_result["insert_count"], "entities to collection: ", collection_name)
    return insert_result

def search(collection_name: str, query_embedding: List, limit: int, output_fields: List[str]) -> List[List[dict]]:
    search_result = milvus_client.search(
        collection_name=collection_name,
        data=query_embedding,
        limit=limit,
        output_fields=output_fields)
    return search_result

def close():
    milvus_client.close()

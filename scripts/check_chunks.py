from app.rag.vector_store import get_client, COLLECTION
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = get_client()
hits = client.scroll(
    collection_name=COLLECTION, 
    scroll_filter=Filter(must=[FieldCondition(key='module', match=MatchValue(value='CARDIOLOGIE'))]), 
    limit=1000, 
    with_payload=True
)[0]

prinzmetal = [h for h in hits if 'prinzmetal' in str(h.payload.get('parent_text', '')).lower()]

for h in prinzmetal:
    print(f"Page: {h.payload.get('page')} | Text: {h.payload.get('parent_text')}")
    print("-" * 40)

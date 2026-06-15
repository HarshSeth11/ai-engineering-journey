def FilterByMetadata(collection):
    # Only search within customer_service documents
    results = collection.query(
        query_texts=["warranty claim process"],
        n_results=2,
        where={"category": "customer_service"}
    )

    print("\nFiltered search (customer_service only):")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['title']}]: {doc[:80]}...")
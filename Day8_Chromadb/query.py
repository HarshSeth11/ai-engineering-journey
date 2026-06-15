def search(query,collection, n_results=2):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # print("Result : ", results)
    # print(" ---------------------------------------------------------------- ")
    
    print(f"\nQuery: '{query}'")
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"  {i+1}. [{metadata['title']}] distance: {distance:.3f}")
        print(f"     {doc[:80]}...")

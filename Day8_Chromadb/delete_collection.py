
def DeleteCollection(collection, client, embedding_fn):

    # Delete the collection object and reload from disk
    del collection

    collection2 = client.get_or_create_collection(
        name="company_policies",
        embedding_function=embedding_fn
    )

    print(f"\n✓ Reloaded from disk — {collection2.count()} documents still there")
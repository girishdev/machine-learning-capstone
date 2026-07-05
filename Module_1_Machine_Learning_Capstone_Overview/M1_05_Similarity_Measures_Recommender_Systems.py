# ============================================================
# SIMILARITY MEASURES FOR RECOMMENDER SYSTEMS
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STEP 1: Define the documents and their contents
# ============================================================

documents = {
    "Document 1": "Political unrest leads to protests.",
    "Document 2": "New technology breakthrough announced.",
    "Document 3": "Team wins championship in a thrilling sports event.",
    "Document 4": "Popular actor's new movie release.",
    "Document 5": "Stock market experiences sharp rise."
}

print("=" * 70)
print("DOCUMENTS:")
print("=" * 70)
for doc_name, content in documents.items():
    print(f"{doc_name}: {content}")

# ============================================================
# STEP 2: Create vocabulary from the documents
# ============================================================

vocabulary = set()
for doc_content in documents.values():
    vocabulary.update(doc_content.lower().split())

vocabulary = sorted(list(vocabulary))

print("\n" + "=" * 70)
print(f"VOCABULARY ({len(vocabulary)} unique words):")
print("=" * 70)
print(vocabulary[:20], "...")

# ============================================================
# STEP 3: Create Bag of Words (BoW) vectors for each document
# ============================================================

bow_vectors = []
for doc_content in documents.values():
    bow_vector = [doc_content.lower().count(word) for word in vocabulary]
    bow_vectors.append(bow_vector)

print("\n" + "=" * 70)
print("BAG OF WORDS VECTORS:")
print("=" * 70)

# Convert BoW vectors to DataFrame
bow_df = pd.DataFrame(bow_vectors, columns=vocabulary, index=documents.keys())

print("\nDocument BoW DataFrame (first 10 words):")
print(bow_df.iloc[:, :10])

# ============================================================
# STEP 4: Define user interests
# ============================================================

user_interests = {
    "User 1": {"politics"},
    "User 2": {"technology", "sports"},
    "User 3": {"entertainment", "finance"}
}

print("\n" + "=" * 70)
print("USER INTERESTS:")
print("=" * 70)
for user, interests in user_interests.items():
    print(f"{user}: {interests}")

# ============================================================
# STEP 5: Create user profiles as BoW vectors
# ============================================================

user_profiles = {}
for user, interests in user_interests.items():
    user_profile = [1 if word in interests else 0 for word in vocabulary]
    user_profiles[user] = user_profile

print("\nUser profile BoW vectors created.")

# Convert user profiles to DataFrame
user_profiles_df = pd.DataFrame(user_profiles, index=vocabulary).T

print("\nUser Profiles DataFrame (first 10 words):")
print(user_profiles_df.iloc[:, :10])

# ============================================================
# STEP 6: Calculate cosine similarity
# ============================================================

print("\n" + "=" * 70)
print("COSINE SIMILARITY CALCULATION:")
print("=" * 70)

similarities = cosine_similarity(user_profiles_df.values, bow_df.values)

# Create DataFrame for similarity scores
similarity_df = pd.DataFrame(
    similarities,
    index=user_profiles_df.index,
    columns=bow_df.index
)

print("\nSimilarity Matrix (User vs Document):")
print(similarity_df)

# ============================================================
# STEP 7: Generate recommendations
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)

recommendations = {}
for user, row in similarity_df.iterrows():
    best_doc_idx = row.argmax()
    best_doc = similarity_df.columns[best_doc_idx]
    best_score = row.max()
    
    recommendations[user] = {
        "document": best_doc,
        "similarity_score": best_score
    }
    
    print(f"\n{user}:")
    print(f"  Recommended Document: {best_doc}")
    print(f"  Similarity Score: {best_score:.4f}")
    print(f"  Document Content: {documents[best_doc]}")

# ============================================================
# STEP 8: Show top recommendations for each user
# ============================================================

print("\n" + "=" * 70)
print("TOP 3 RECOMMENDATIONS FOR EACH USER:")
print("=" * 70)

for user, row in similarity_df.iterrows():
    top_3_indices = row.nlargest(3).index
    top_3_scores = row.nlargest(3).values
    
    print(f"\n{user}:")
    for i, (doc, score) in enumerate(zip(top_3_indices, top_3_scores), 1):
        print(f"  {i}. {doc} (Similarity: {score:.4f})")

# ============================================================
# STEP 9: Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print(f"Total Documents: {len(documents)}")
print(f"Total Users: {len(user_profiles)}")
print(f"Vocabulary Size: {len(vocabulary)} words")
print(f"Similarity Matrix Shape: {similarity_df.shape}")
print("\nRecommender System Analysis Complete!")
print("=" * 70)

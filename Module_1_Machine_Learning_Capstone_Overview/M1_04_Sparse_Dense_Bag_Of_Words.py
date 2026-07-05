# ============================================================
# SPARSE vs DENSE BAG OF WORDS REPRESENTATION
# ============================================================

# Import required libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ============================================================
# PART 1 - SPARSE BAG OF WORDS REPRESENTATION
# ============================================================

print("=" * 70)
print("SPARSE BAG OF WORDS REPRESENTATION (CountVectorizer)")
print("=" * 70)

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create CountVectorizer for sparse BoW
vectorizer = CountVectorizer()

# Fit and transform documents to sparse BoW matrix
sparse_bow = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

print("\nSparse BoW Matrix (as dense array):")
print(sparse_bow.toarray())

print("\nFeature Names (Words):")
print(list(feature_names))

print("\nMatrix Shape (documents, unique_words):", sparse_bow.shape)
print("Data Type: Sparse Matrix (CSR format)")
print("Non-zero elements:", sparse_bow.nnz)
print("Sparsity (% of zeros):", f"{(1 - sparse_bow.nnz / (sparse_bow.shape[0] * sparse_bow.shape[1])) * 100:.2f}%")

# ============================================================
# PART 2 - DENSE BAG OF WORDS USING TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("DENSE BAG OF WORDS REPRESENTATION (TF-IDF Vectorizer)")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer()
dense_bow_tfidf = tfidf_vectorizer.fit_transform(documents)

print("\nTF-IDF Matrix (Normalized - Dense representation):")
print(dense_bow_tfidf.toarray())

print("\nFeature Names:")
print(list(tfidf_vectorizer.get_feature_names_out()))

print("\nMatrix Shape:", dense_bow_tfidf.shape)
print("Data Type: TF-IDF weighted matrix (values between 0-1)")

# ============================================================
# PART 3 - MANUAL DENSE WORD VECTORS
# ============================================================

print("\n" + "=" * 70)
print("MANUAL DENSE WORD VECTORS (Custom Embeddings)")
print("=" * 70)

# Create a simple manual embedding dictionary for demonstration
manual_embeddings = {
    "first": np.array([0.25, 0.45, 0.12, -0.33, 0.78]),
    "document": np.array([0.88, -0.12, 0.45, 0.23, 0.01]),
    "second": np.array([-0.15, 0.67, 0.34, 0.12, 0.89]),
    "third": np.array([0.45, 0.23, -0.56, 0.78, 0.12]),
    "is": np.array([0.12, -0.34, 0.56, 0.11, 0.22]),
    "this": np.array([0.33, 0.44, 0.55, -0.12, 0.99]),
}

words = ["first", "document", "second", "third"]

print("\nCustom Dense Word Vectors (5 dimensions each):")
for word in words:
    if word in manual_embeddings:
        print(f"\nWord: '{word}'")
        print(f"Vector: {manual_embeddings[word]}")
        print(f"Vector Shape: {manual_embeddings[word].shape}")
    else:
        print(f"\nWord: '{word}' - Not found in vocabulary")

# ============================================================
# PART 4 - COMPARISON AND SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON: SPARSE vs DENSE REPRESENTATIONS")
print("=" * 70)

comparison_text = """
╔═══════════════════════════════════════════════════════════════════╗
║ FEATURE                 │ SPARSE BoW          │ DENSE VECTORS     ║
╠═════════════════════════════════════════════════════════════════════╣
║ Representation          │ Count/Binary values │ Continuous floats  ║
║ Dimensionality          │ High (vocab size)   │ Fixed (e.g., 300)  ║
║ Sparsity                │ High (many zeros)   │ Low (mostly filled) ║
║ Example values          │ [0,1,2,0,1,...]     │ [0.23,-0.45,0.12]  ║
║ Memory efficient        │ YES (sparse format) │ NO (dense matrix)   ║
║ Semantic meaning        │ Weak                │ Strong              ║
║ Use in NLP              │ Text classification │ Deep learning       ║
║ Interpretability        │ High                │ Low                 ║
║ Computational cost      │ Low                 │ Medium-High         ║
╚═══════════════════════════════════════════════════════════════════════╝

SPARSE BAG OF WORDS (CountVectorizer):
  • Creates a matrix with dimensions: (n_documents, n_unique_words)
  • Each cell contains the COUNT of how many times a word appears
  • Example: Document [0, 1, 2, 1, 0, 3, ...]
  • Memory: Very efficient using sparse matrix format (CSR)
  • Best for: Text classification, spam detection, keyword analysis

DENSE WORD VECTORS (Embeddings):
  • Creates fixed-size vectors for each word
  • Each vector has meaningful dimensions representing semantic features
  • Example: "king" [0.23, -0.45, 0.12, ..., 0.87]
  • Memory: Fixed size per vector, scales with dimension count
  • Best for: Word similarity, neural networks, semantic analysis
  • Can capture relationships (king - man + woman ≈ queen)
"""

print(comparison_text)

# ============================================================
# FINAL STATISTICS
# ============================================================

print("=" * 70)
print("DATA STATISTICS")
print("=" * 70)

print(f"\nNumber of documents: {len(documents)}")
print(f"Number of unique words (Sparse): {len(feature_names)}")
print(f"Sparse BoW matrix shape: {sparse_bow.shape}")
print(f"Sparse matrix density: {(sparse_bow.nnz / (sparse_bow.shape[0] * sparse_bow.shape[1])) * 100:.2f}%")
print(f"TF-IDF matrix shape: {dense_bow_tfidf.shape}")
print(f"Dense vector dimension: {manual_embeddings['first'].shape[0]}")
print(f"\nTotal sparse values: {sparse_bow.nnz}")
print(f"Total zero values in sparse matrix: {(sparse_bow.shape[0] * sparse_bow.shape[1]) - sparse_bow.nnz}")

print("\n" + "=" * 70)
print("✓ Sparse vs Dense BoW Analysis Complete!")
print("=" * 70)





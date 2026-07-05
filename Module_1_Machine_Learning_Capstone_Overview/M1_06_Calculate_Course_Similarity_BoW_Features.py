# ============================================================
# CALCULATE COURSE SIMILARITY USING BAG OF WORDS FEATURES
# ============================================================
# IBM Machine Learning Capstone Project
# Similarity measurement between courses using BoW features
# ============================================================

# ============================================================
# STEP 1: Install and Import Required Libraries
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk

from scipy.spatial.distance import cosine, euclidean

print("=" * 80)
print("CALCULATING COURSE SIMILARITY USING BAG OF WORDS FEATURES")
print("=" * 80)

# Set random state
rs = 123

# ============================================================
# STEP 2: Example 1 - Calculate Cosine Similarity Between Two Simple Courses
# ============================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: SIMPLE COURSE SIMILARITY")
print("=" * 80)

course1 = "machine learning for everyone"
course2 = "machine learning for beginners"

print(f"\nCourse 1: {course1}")
print(f"Course 2: {course2}")

# Create vocabulary
tokens = set(course1.split() + course2.split())
tokens = sorted(list(tokens))

print(f"\nVocabulary ({len(tokens)} words): {tokens}")

# ============================================================
# STEP 3: Define Function to Generate Sparse Bag of Words
# ============================================================

def generate_sparse_bow(course, vocabulary):
    """
    Generate a sparse bag-of-words (BoW) representation for a given course.

    Parameters:
    course (str): The input course text to generate the BoW representation for.
    vocabulary (list): List of unique words to use as features.

    Returns:
    list: A sparse BoW representation where each element corresponds to the count
    of a word in the input course text.
    """
    # Initialize an empty list to store the BoW vector
    bow_vector = []

    # Tokenize the course text by splitting it into words
    words = course.lower().split()

    # Iterate through all unique words (tokens) in the vocabulary
    for token in vocabulary:
        # Count how many times the token appears in the course
        count = words.count(token)
        # Append the count to the BoW vector
        bow_vector.append(count)

    # Return the sparse BoW vector
    return bow_vector

# Generate BoW vectors for both courses
bow1 = generate_sparse_bow(course1, tokens)
bow2 = generate_sparse_bow(course2, tokens)

print(f"\nBoW Vector for Course 1: {bow1}")
print(f"BoW Vector for Course 2: {bow2}")

# ============================================================
# STEP 4: Calculate Different Similarity Measures
# ============================================================

# Calculate Cosine Similarity
cos_sim = 1 - cosine(bow1, bow2)
print(f"\nCosine Similarity: {round(cos_sim, 4)} ({round(cos_sim, 2) * 100}%)")

# Calculate Euclidean Distance
euclidean_dist = euclidean(bow1, bow2)
print(f"Euclidean Distance: {round(euclidean_dist, 4)}")

# ============================================================
# STEP 5: Load Real Course Data and BoW Features
# ============================================================

print("\n" + "=" * 80)
print("LOADING REAL COURSE DATA")
print("=" * 80)

# Load the BoW features as Pandas dataframe
bows_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/courses_bows.csv"
print(f"\nLoading BoW features from: {bows_url}")

try:
    bows_df = pd.read_csv(bows_url)
    bows_df = bows_df[['doc_id', 'token', 'bow']]
    print(f"✓ BoW features loaded successfully ({len(bows_df)} rows)")
    print("\nFirst 10 rows of BoW DataFrame:")
    print(bows_df.head(10))
except Exception as e:
    print(f"✗ Error loading BoW features: {e}")
    print("Continuing with example data only...")
    bows_df = None

# Load the course dataframe
course_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
print(f"\nLoading course data from: {course_url}")

try:
    course_df = pd.read_csv(course_url)
    print(f"✓ Course data loaded successfully ({len(course_df)} courses)")
    print("\nFirst 10 rows of Course DataFrame:")
    print(course_df.head(10))
except Exception as e:
    print(f"✗ Error loading course data: {e}")
    print("Continuing without course data...")
    course_df = None

# ============================================================
# STEP 6: Define Function to Pivot Two BoW Vectors
# ============================================================

def pivot_two_bows(basedoc, comparedoc):
    """
    Pivot two bag-of-words (BoW) representations for comparison.

    Parameters:
    basedoc (DataFrame): DataFrame containing BoW for the base document.
    comparedoc (DataFrame): DataFrame containing BoW for the document to compare.

    Returns:
    DataFrame: Pivoted BoW representations facilitating direct comparison.
    """
    # Create copies of the input DataFrames to avoid modifying the originals
    base = basedoc.copy()
    base['type'] = 'base'  # Add a 'type' column indicating base document
    compare = comparedoc.copy()
    compare['type'] = 'compare'  # Add a 'type' column indicating compared document

    # Concatenate the two DataFrames vertically
    join = pd.concat([base, compare])

    # Pivot the concatenated DataFrame
    joinT = join.pivot(index=['doc_id', 'type'], columns='token').fillna(0).reset_index(level=[0, 1])

    # Assign meaningful column names to the pivoted DataFrame
    joinT.columns = ['doc_id', 'type'] + [t[1] for t in joinT.columns][2:]

    # Return the pivoted DataFrame for comparison
    return joinT

# ============================================================
# STEP 7: Example 2 - Calculate Similarity Between Two Real Courses
# ============================================================

if bows_df is not None and course_df is not None:
    print("\n" + "=" * 80)
    print("EXAMPLE 2: REAL COURSE SIMILARITY")
    print("=" * 80)
    
    course1_id = 'ML0151EN'
    course2_id = 'ML0101ENv3'
    
    print(f"\nComparing courses: {course1_id} and {course2_id}")
    
    # Get course information
    course1_info = course_df[course_df['COURSE_ID'] == course1_id]
    course2_info = course_df[course_df['COURSE_ID'] == course2_id]
    
    if not course1_info.empty:
        print(f"\n{course1_id}: {course1_info.iloc[0]['TITLE']}")
    if not course2_info.empty:
        print(f"{course2_id}: {course2_info.iloc[0]['TITLE']}")
    
    # Get BoW data
    course1_bow = bows_df[bows_df['doc_id'] == course1_id]
    course2_bow = bows_df[bows_df['doc_id'] == course2_id]
    
    if not course1_bow.empty and not course2_bow.empty:
        # Pivot the vectors
        bow_vectors = pivot_two_bows(course1_bow, course2_bow)
        
        print("\nBoW Vectors (pivoted):")
        print(bow_vectors.iloc[:, :10])  # Show first 10 columns
        
        # Calculate similarity
        similarity = 1 - cosine(bow_vectors.iloc[0, 2:], bow_vectors.iloc[1, 2:])
        print(f"\nCosine Similarity: {round(similarity, 4)}")

# ============================================================
# STEP 8: MAIN TASK - Find Similar Courses
# ============================================================

if bows_df is not None and course_df is not None:
    print("\n" + "=" * 80)
    print("FINDING COURSES SIMILAR TO 'MACHINE LEARNING WITH PYTHON'")
    print("=" * 80)
    
    # Base course
    base_course_id = "ML0101ENv3"
    
    base_course_info = course_df[course_df['COURSE_ID'] == base_course_id]
    if not base_course_info.empty:
        print(f"\nBase Course: {base_course_id}")
        print(f"Title: {base_course_info.iloc[0]['TITLE']}")
    
    # Get BoW for base course
    base_course = bows_df[bows_df['doc_id'] == base_course_id]
    
    # Similarity threshold
    threshold = 0.5
    print(f"Similarity Threshold: {threshold}")
    
    similar_courses = []
    
    # Iterate through all course ids
    print("\nCalculating similarities...")
    
    for course_id in bows_df['doc_id'].unique():
        
        # Skip self comparison
        if course_id == base_course_id:
            continue
        
        compare_course = bows_df[bows_df['doc_id'] == course_id]
        
        # Create comparable vectors
        bow_vectors = pivot_two_bows(base_course, compare_course)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(
            bow_vectors.iloc[0, 2:],
            bow_vectors.iloc[1, 2:]
        )
        
        # Save if above threshold
        if similarity > threshold:
            similar_courses.append((course_id, similarity))
    
    # Sort descending by similarity
    similar_courses = sorted(similar_courses, key=lambda x: x[1], reverse=True)
    
    # ============================================================
    # STEP 9: Display Results
    # ============================================================
    
    print("\n" + "=" * 80)
    print(f"SIMILAR COURSES (Threshold: {threshold})")
    print("=" * 80)
    
    if similar_courses:
        print(f"\nFound {len(similar_courses)} similar courses:\n")
        
        for cid, sim in similar_courses:
            course_info = course_df[course_df['COURSE_ID'] == cid]
            
            if not course_info.empty:
                title = course_info.iloc[0]['TITLE']
                description = course_info.iloc[0]['DESCRIPTION']
                
                print("=" * 80)
                print(f"Course ID      : {cid}")
                print(f"Similarity     : {round(sim, 4)} ({round(sim, 2) * 100}%)")
                print(f"Title          : {title}")
                print(f"Description    : {description[:150]}...")
                print("=" * 80 + "\n")
    else:
        print(f"\nNo courses found with similarity > {threshold}")

# ============================================================
# STEP 10: Summary
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
This script demonstrates how to calculate course similarity using Bag of Words (BoW)
features and cosine similarity measurement.

Key Concepts:
1. Bag of Words (BoW): Represents text as word counts/presence
2. Cosine Similarity: Measures angle between two vectors (0 to 1)
3. Euclidean Distance: Measures straight-line distance between vectors
4. Threshold: Similarity score (0-1) to determine if courses are similar

Applications:
- Content-based recommender systems
- Finding similar courses for users
- Course clustering and categorization
- Personalized learning path recommendations
""")
print("=" * 80)
print("✓ Script Execution Complete!")
print("=" * 80)

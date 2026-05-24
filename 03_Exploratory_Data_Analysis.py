# ============================================================
# IBM MACHINE LEARNING CAPSTONE PROJECT
# Bag of Words (BoW) Feature Extraction
# ============================================================

# ============================================================
# STEP 1 - INSTALL REQUIRED LIBRARIES
# ============================================================

# Uncomment these lines if libraries are not installed

# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install seaborn
# !pip install scikit-learn
# !pip install wordcloud


# ============================================================
# STEP 2 - IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud


# ============================================================
# STEP 3 - LOAD THE DATASET
# ============================================================

# Replace this with your actual dataset path or URL

course_df = pd.read_csv(course_genre_url)

# Display first 5 rows
print("First 5 Rows:")
print(course_df.head())


# ============================================================
# STEP 4 - CHECK DATASET INFORMATION
# ============================================================

print("\nDataset Shape:")
print(course_df.shape)

print("\nDataset Columns:")
print(course_df.columns)


# ============================================================
# STEP 5 - DISPLAY COURSE TITLES
# ============================================================

print("\nCourse Titles:")
print(course_df['TITLE'].head())


# ============================================================
# STEP 6 - CONVERT COURSE TITLES TO LIST
# ============================================================

# Convert course titles into list format

documents = course_df['TITLE'].astype(str).tolist()

print("\nTotal Documents:")
print(len(documents))


# ============================================================
# STEP 7 - INITIALIZE COUNT VECTORIZER
# ============================================================

# stop_words='english'
# removes common English words like:
# the, is, and, of, etc.

vectorizer = CountVectorizer(
    stop_words='english'
)


# ============================================================
# STEP 8 - CREATE BAG OF WORDS MATRIX
# ============================================================

# Learn vocabulary and transform text into vectors

bow_matrix = vectorizer.fit_transform(documents)

print("\nBoW Matrix Created Successfully")


# ============================================================
# STEP 9 - CHECK BOW MATRIX SHAPE
# ============================================================

print("\nBoW Matrix Shape:")
print(bow_matrix.shape)

# Meaning:
# Rows    -> Number of courses
# Columns -> Number of unique words


# ============================================================
# STEP 10 - EXTRACT FEATURE NAMES
# ============================================================

feature_names = vectorizer.get_feature_names_out()

print("\nFeature Names:")
print(feature_names[:20])


# ============================================================
# STEP 11 - CONVERT SPARSE MATRIX TO DATAFRAME
# ============================================================

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=feature_names
)

print("\nBoW DataFrame:")
print(bow_df.head())


# ============================================================
# STEP 12 - ADD COURSE TITLES BACK
# ============================================================

bow_df['TITLE'] = course_df['TITLE']

print("\nBoW DataFrame with Titles:")
print(bow_df.head())


# ============================================================
# STEP 13 - CHECK WORD FREQUENCY
# ============================================================

# Example:
# Check frequency of word "python"

if 'python' in bow_df.columns:

    print("\nPython Word Frequency:")
    print(bow_df[['python', 'TITLE']].head())

else:
    print("\nWord 'python' not found in vocabulary")


# ============================================================
# STEP 14 - CALCULATE TOTAL WORD COUNTS
# ============================================================

# Remove TITLE column before summing

word_features = bow_df.drop(columns=['TITLE'])

word_counts = word_features.sum().sort_values(
    ascending=False
)

print("\nTop 20 Most Frequent Words:")
print(word_counts.head(20))


# ============================================================
# STEP 15 - VISUALIZE TOP WORDS
# ============================================================

top_words = word_counts.head(20)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_words.index,
    y=top_words.values
)

plt.xticks(rotation=90)

plt.title("Top 20 Most Common Words")

plt.xlabel("Words")

plt.ylabel("Frequency")

plt.show()


# ============================================================
# STEP 16 - GENERATE WORDCLOUD
# ============================================================

# Combine all course titles into one string

text = " ".join(documents)

wordcloud = WordCloud(
    background_color='white',
    width=1000,
    height=500
).generate(text)

plt.figure(figsize=(15, 8))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Course Title WordCloud")

plt.show()


# ============================================================
# STEP 17 - CALCULATE COSINE SIMILARITY
# ============================================================

# This calculates similarity between courses

similarity_matrix = cosine_similarity(
    bow_matrix
)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)


# ============================================================
# STEP 18 - FIND SIMILAR COURSES
# ============================================================

# Example:
# Find courses similar to first course

similar_courses = list(
    enumerate(similarity_matrix[0])
)

print("\nSimilar Courses:")
print(similar_courses[:10])


# ============================================================
# STEP 19 - SORT SIMILAR COURSES
# ============================================================

sorted_courses = sorted(
    similar_courses,
    key=lambda x: x[1],
    reverse=True
)

print("\nSorted Similar Courses:")
print(sorted_courses[:10])


# ============================================================
# STEP 20 - DISPLAY RECOMMENDED COURSES
# ============================================================

print("\nRecommended Courses:\n")

for course in sorted_courses[1:6]:

    index = course[0]

    title = course_df.iloc[index]['TITLE']

    similarity_score = course[1]

    print(f"Course: {title}")

    print(f"Similarity Score: {similarity_score}")

    print("-" * 50)


# ============================================================
# STEP 21 - SAVE BOW FEATURES TO CSV
# ============================================================

bow_df.to_csv(
    "bow_features.csv",
    index=False
)

print("\nBoW Features Saved Successfully")


# ============================================================
# STEP 22 - FINAL SUMMARY
# ============================================================

print("\n========== PROJECT SUMMARY ==========")

print("1. Loaded Course Dataset")

print("2. Extracted Course Titles")

print("3. Created Bag of Words Features")

print("4. Calculated Word Frequencies")

print("5. Generated WordCloud")

print("6. Calculated Course Similarity")

print("7. Generated Basic Recommendations")

print("8. Saved Features to CSV")

print("====================================")
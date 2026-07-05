"""
================================================================================
EXPLORATORY DATA ANALYSIS ON ONLINE COURSE ENROLLMENT DATA
================================================================================

This script performs comprehensive exploratory data analysis on online courses
related datasets such as course titles/genres and course enrollments.

Objectives:
    - Identify keywords in course titles using a WordCloud
    - Calculate summary statistics of the online course content dataset
    - Determine popular course genres
    - Calculate summary statistics of course enrollments
    - Identify courses with the greatest number of enrolled students

Author: Yan Luo
Updated: 2024
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Fix encoding issues for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set random state for reproducibility
rs = 123

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Point to the datasets stored on the cloud
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"

# Load datasets
try:
    course_df = pd.read_csv(course_genre_url)
    ratings_df = pd.read_csv(ratings_url)
    print(f"[OK] Course data loaded: {course_df.shape[0]} courses")
    print(f"[OK] Ratings data loaded: {ratings_df.shape[0]} enrollments")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("\n" + "="*80)
print("COURSE DATASET EXPLORATION")
print("="*80)

# Check columns
print("\nCourse DataFrame Columns:")
print(course_df.columns.tolist())

# Check data types
print("\nCourse DataFrame Data Types:")
print(course_df.dtypes)

# Check shape
print(f"\nTotal number of courses: {course_df.shape[0]}")

# Display first few rows
print("\nFirst 5 courses:")
print(course_df.head())

# Check a specific course
print("\nExample course (index 1):")
print(course_df.iloc[1])

print("\n" + "="*80)
print("WORDCLOUD ANALYSIS - COURSE TITLES")
print("="*80)

# Join all titles into one string
titles = " ".join(title for title in course_df['TITLE'].astype(str))
print(f"Total title string length: {len(titles)} characters")

# Create stopwords set
stopwords = set(STOPWORDS)
stopwords.update(["getting started", "using", "enabling", "template", "university", "end", "introduction", "basic"])

# Generate WordCloud
print("\nGenerating WordCloud...")
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400)
wordcloud.generate(titles)

# Visualize WordCloud
plt.figure(figsize=(40, 20))
plt.axis("off")
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Course Keywords WordCloud", fontsize=20, fontweight='bold')
plt.savefig('wordcloud.png', dpi=100, bbox_inches='tight')
print("WordCloud saved as 'wordcloud.png'")
plt.close()

print("[OK] WordCloud generated successfully")
print("\nKey insights from WordCloud:")
print("- Popular IT keywords: python, data science, machine learning, big data, ai, tensorflow")
print("- Focus on demanding IT skills in courses")

print("\n" + "="*80)
print("COURSE GENRE ANALYSIS")
print("="*80)

# Find Machine Learning courses
print("\n>>> Finding Machine Learning courses...")
ml_courses = course_df[course_df['MachineLearning'] == 1]
print(f"Found {len(ml_courses)} Machine Learning courses")
print(ml_courses[['COURSE_ID', 'TITLE']].head(10))

# Find Machine Learning + Big Data courses
print("\n>>> Finding Machine Learning + Big Data courses...")
ml_bigdata_courses = course_df[
    (course_df['MachineLearning'] == 1) &
    (course_df['BigData'] == 1)
]
print(f"Found {len(ml_bigdata_courses)} courses with both Machine Learning and Big Data genres")
print(ml_bigdata_courses[['COURSE_ID', 'TITLE']].head(10))

# Calculate course count per genre
print("\n>>> Calculating course count per genre...")
genres = course_df.columns[2:]
genre_sums = course_df[genres].sum(axis=0)
genre_counts = pd.DataFrame(
    genre_sums,
    columns=['Count']
)
genre_counts = genre_counts.sort_values(by='Count', ascending=False)

print("\nTop 10 Course Genres:")
print(genre_counts.head(10))

# Visualize genre counts
print("\nPlotting genre counts...")
plt.figure(figsize=(12, 6))
plot = sns.barplot(
    x=genre_counts.index,
    y='Count',
    data=genre_counts
)
plt.xticks(rotation=90)
plt.title("Course Genre Counts", fontsize=14, fontweight='bold')
plt.ylabel("Number of Courses")
plt.xlabel("Genre")
plt.tight_layout()
plt.savefig('genre_counts.png', dpi=100, bbox_inches='tight')
print("Genre counts chart saved as 'genre_counts.png'")
plt.close()

print("[OK] Genre analysis completed")

print("\n" + "="*80)
print("RATINGS DATASET EXPLORATION")
print("="*80)

# Check ratings data
print("\nRatings DataFrame shape:", ratings_df.shape)
print("\nFirst 5 ratings records:")
print(ratings_df.head())

# Check unique ratings
print("\nUnique rating values:")
print(ratings_df['rating'].unique())

print("\nRating scale explanation:")
print("- Rating 5: Excellent - Highly recommended")
print("- Rating 4: Good - Recommended with minor improvements")
print("- Rating 3: Below expectations - Needs significant modifications")

print(f"\nTotal number of ratings: {ratings_df.shape[0]}")

print("\n" + "="*80)
print("USER ENROLLMENT ANALYSIS")
print("="*80)

# Aggregate ratings per user
print("\n>>> Calculating user enrollment counts...")
user_rating_counts = ratings_df.groupby('user').size().reset_index(name='count')
print(f"Total number of users: {len(user_rating_counts)}")

print("\nUser Enrollment Statistics:")
print(user_rating_counts['count'].describe())

# Visualize user enrollment distribution
print("\nPlotting user enrollment distribution...")
plt.figure(figsize=(10, 6))
user_rating_counts['count'].hist(bins=50, edgecolor='black')
plt.title("User Enrollment Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Number of Ratings per User")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig('user_enrollment_distribution.png', dpi=100, bbox_inches='tight')
print("User enrollment distribution saved as 'user_enrollment_distribution.png'")
plt.close()

print("[OK] User enrollment analysis completed")

print("\n" + "="*80)
print("TOP 20 MOST POPULAR COURSES")
print("="*80)

# Find top 20 courses by rating count
print("\n>>> Finding top 20 most popular courses...")
top_courses = ratings_df.groupby('item').size().reset_index(name='Ratings')
top_courses = top_courses.sort_values(by='Ratings', ascending=False).head(20)

# Merge with course metadata to get titles
top_courses = pd.merge(
    top_courses,
    course_df[['COURSE_ID', 'TITLE']],
    how='left',
    left_on='item',
    right_on='COURSE_ID'
)

print("\nTop 20 Most Popular Courses:")
print(top_courses[['TITLE', 'Ratings']].to_string(index=False))

# Calculate percentage of top 20 courses
print("\n>>> Calculating enrollment concentration...")
total_enrollments = ratings_df.shape[0]
top_20_enrollments = top_courses['Ratings'].sum()
percentage = round((top_20_enrollments * 100) / total_enrollments, 2)

print(f"\nTotal enrollments in dataset: {total_enrollments}")
print(f"Enrollments for top 20 courses: {top_20_enrollments}")
print(f"Percentage of top 20 course enrollments: {percentage}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Dataset Overview:
- Total Courses: {course_df.shape[0]}
- Total Course Genres: {len(genres)}
- Total User Enrollments: {ratings_df.shape[0]}
- Total Unique Users: {len(user_rating_counts)}
- Average Enrollments per User: {user_rating_counts['count'].mean():.2f}
- Average User Count per Course: {ratings_df.groupby('item').size().mean():.2f}

Key Findings:
1. Top 3 Course Genres: {', '.join(genre_counts.head(3).index.tolist())}
2. Machine Learning Courses: {len(ml_courses)}
3. ML + Big Data Courses: {len(ml_bigdata_courses)}
4. Top 20 Courses Account For: {percentage}% of all enrollments
5. Most Active User Ratings: {user_rating_counts['count'].max()} courses
6. Least Active User Ratings: {user_rating_counts['count'].min()} course(s)

This analysis provides preliminary understanding of course metadata and enrollments,
which will be used for building a recommender system in later modules.
""")

print("="*80)
print("[OK] Exploratory Data Analysis Complete!")
print("="*80 + "\n")

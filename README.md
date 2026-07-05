# Machine Learning Capstone: Personalized Course Recommender System

## Project Overview

This project is part of the **Machine Learning Capstone** and focuses on building a **personalized course recommender system** for an online learning platform. The main goal of the project is to recommend relevant courses to learners based on course content, user interaction history, course genres, ratings, and learner behavior patterns.

Online learning platforms contain a large number of courses, which makes it difficult for learners to identify the most suitable course for their interests and career goals. This project solves that problem by applying multiple machine learning-based recommendation techniques, including content-based filtering, clustering, collaborative filtering, neural network embeddings, regression, and classification models.

---

## Business Problem

Learners often face difficulty in selecting the right course from a large course catalog. A personalized recommender system can improve the learning experience by suggesting relevant courses based on:

* Learner interests
* Previously completed or viewed courses
* Course genres and descriptions
* Similar learner behavior
* Course popularity
* Predicted rating or preference score

This project aims to improve course discovery, increase learner engagement, and support personalized learning paths.

---

## Objectives

The main objectives of this project are:

* Perform exploratory data analysis on course and user interaction data.
* Analyze course genre distribution and learner enrollment behavior.
* Identify the most popular courses.
* Generate course title word clouds to understand common course topics.
* Build content-based recommender systems using course genres and course similarity.
* Build clustering-based recommendations using PCA and KMeans.
* Build collaborative filtering models using KNN and NMF.
* Build a neural network-based recommender using user and course embeddings.
* Predict course ratings using regression and classification models.
* Compare different recommendation approaches and summarize key findings.

---

## Dataset Description

The project uses course and user interaction datasets containing information such as:

* Course ID
* Course title
* Course description
* Course genre/category
* User ID
* User-course ratings or enrollments
* User-course interaction records

These datasets are used to analyze learner behavior and build different recommendation models.

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Surprise Library
* TensorFlow / Keras
* Gensim
* WordCloud

---

## Machine Learning Concepts Covered

This project covers the following machine learning and data science concepts:

* Exploratory Data Analysis
* Data visualization
* Text preprocessing
* Tokenization
* Stop-word removal
* Bag of Words feature engineering
* Cosine similarity
* Content-based filtering
* User profile-based recommendation
* Course similarity recommendation
* PCA dimensionality reduction
* KMeans clustering
* Collaborative filtering
* KNN-based recommendation
* NMF matrix factorization
* Neural network embeddings
* Regression-based rating prediction
* Classification-based rating prediction
* Model evaluation and comparison

---

## Project Workflow

```text
Course and User Interaction Data
        ↓
Exploratory Data Analysis
        ↓
Text Preprocessing and Feature Engineering
        ↓
Bag of Words and Course Similarity
        ↓
Content-Based Recommendation
        ↓
Clustering-Based Recommendation
        ↓
Collaborative Filtering using KNN and NMF
        ↓
Neural Network-Based Recommendation
        ↓
Regression and Classification Rating Prediction
        ↓
Model Evaluation and Comparison
        ↓
Final Recommendation System
```

---

## Exploratory Data Analysis

The EDA phase helped understand the course catalog and learner behavior. The following analyses were performed:

* Course counts per genre
* User enrollment distribution
* Top 20 most popular courses
* Course keyword analysis using WordCloud
* Course popularity analysis
* Learner activity pattern analysis

### Key EDA Findings

* Python, Data Science, Machine Learning, Cloud, and Big Data are highly common course topics.
* Some courses receive significantly higher learner enrollments than others.
* Most learners enroll in only a few courses.
* Course genre distribution helps understand the structure of the course catalog.

---

## Recommendation Approaches

## 1. User Profile-Based Recommender

This approach recommends courses based on a learner’s previous course history and course genre preferences.

### Workflow

```text
Completed Courses
        ↓
Course Genre Matrix
        ↓
Build User Profile Vector
        ↓
Calculate Recommendation Scores
        ↓
Rank Unseen Courses
        ↓
Recommend Top-N Courses
```

### Strengths

* Personalized and easy to interpret
* Fast recommendation generation
* Useful when course genre information is available

### Limitations

* Cold start problem for new users
* Depends on accurate course genre labels
* May recommend similar topics repeatedly

---

## 2. Course Similarity-Based Recommender

This approach recommends courses similar to the courses a learner has already viewed or completed.

### Workflow

```text
Course Descriptions
        ↓
Text Preprocessing
        ↓
Bag of Words Features
        ↓
Cosine Similarity Matrix
        ↓
Top-N Similar Course Recommendations
```

### Evaluation Summary

| Evaluation Area   | Result                                                         |
| ----------------- | -------------------------------------------------------------- |
| Feature Type      | Bag of Words course text features                              |
| Similarity Method | Cosine Similarity                                              |
| Input             | User’s completed or viewed courses                             |
| Output            | Top-N similar course recommendations                           |
| Strength          | Recommends closely related technical courses                   |
| Limitation        | May reduce diversity by recommending similar topics repeatedly |

---

## 3. Clustering-Based Recommender

This approach groups similar learners using PCA and KMeans clustering. After grouping learners, the system recommends popular unseen courses from the same cluster.

### Workflow

```text
User-Course Interaction Data
        ↓
PCA Dimensionality Reduction
        ↓
KMeans Clustering
        ↓
Learner Groups / Clusters
        ↓
Find Popular Courses in Same Cluster
        ↓
Remove Already Enrolled Courses
        ↓
Recommend Unseen Courses
```

### Strengths

* Useful for learner segmentation
* Scalable for large user groups
* Recommends based on similar learner behavior

### Limitations

* Depends on selecting the right number of clusters
* Recommendations may be less personalized than individual models

---

## 4. KNN-Based Collaborative Filtering

KNN collaborative filtering recommends courses by finding learners with similar course preferences.

### Workflow

```text
User-Course Rating Matrix
        ↓
Target User
        ↓
Calculate Similarity with Other Users
        ↓
Select K Nearest Neighbors
        ↓
Aggregate Neighbor Ratings
        ↓
Predict Ratings for Unseen Courses
        ↓
Recommend Top-N Courses
```

### Evaluation Metric

* RMSE

Lower RMSE indicates better rating prediction performance.

---

## 5. NMF-Based Collaborative Filtering

NMF, or Non-negative Matrix Factorization, decomposes the user-course rating matrix into hidden user and course features.

### Workflow

```text
User-Course Rating Matrix
        ↓
Apply Non-negative Matrix Factorization
        ↓
Learn User Latent Features
        ↓
Learn Course Latent Features
        ↓
Reconstruct Predicted Rating Matrix
        ↓
Recommend Top-N Courses
```

### Strengths

* Handles sparse user-course interaction data effectively
* Learns hidden relationships between learners and courses
* Useful for rating prediction

---

## 6. Neural Network-Based Recommender

The neural network recommender uses user embeddings and course embeddings to learn hidden user-course relationships.

### Workflow

```text
User ID
        ↓
User Embedding Layer

Course ID
        ↓
Course Embedding Layer

User Embedding + Course Embedding
        ↓
Interaction / Concatenation Layer
        ↓
Dense Neural Network Layers
        ↓
Predicted Rating / Preference Score
        ↓
Course Recommendation
```

### Strengths

* Learns complex non-linear relationships
* Captures hidden user-course patterns
* Provides stronger personalization when enough data is available

---

## 7. Rating Prediction

The project also includes rating prediction using regression and classification models.

### Regression Models

Regression models predict numerical rating scores.

Models used:

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet

Evaluation metric:

* RMSE

### Classification Models

Classification models predict rating categories or rating modes.

Models used:

* Logistic Regression
* Random Forest

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score

---

## Model Comparison

| Model                    | Main Idea                                  | Evaluation Metric               | Strength                                          |
| ------------------------ | ------------------------------------------ | ------------------------------- | ------------------------------------------------- |
| User Profile Recommender | Matches learner profile with course genres | Recommendation Score            | Simple and explainable                            |
| Course Similarity        | Finds similar courses using text features  | Cosine Similarity               | Fast and useful for related course recommendation |
| Clustering               | Groups similar learners                    | Elbow Method / Cluster Quality  | Useful for learner segmentation                   |
| KNN                      | Finds similar users                        | RMSE                            | Simple and interpretable                          |
| NMF                      | Matrix factorization                       | RMSE                            | Handles sparse data effectively                   |
| Neural Network           | Learns embeddings                          | Loss / RMSE                     | Captures hidden user-course relationships         |
| Regression               | Predicts rating scores                     | RMSE                            | Useful for numerical rating prediction            |
| Classification           | Predicts rating categories                 | Accuracy, Precision, Recall, F1 | Useful for rating class prediction                |

---

## Key Findings

* Course recommendation improves learner experience by reducing manual search effort.
* Content-based methods are simple, fast, and explainable.
* Course similarity works well when course descriptions share common technical keywords.
* User profile-based recommendation is useful when learner history is available.
* Clustering helps group similar learners and recommend popular unseen courses.
* KNN is easy to understand and useful as a collaborative filtering baseline.
* NMF is effective for sparse user-course interaction data.
* Neural network embeddings can capture deeper relationships between users and courses.
* Regression and classification models help predict learner preferences.
* A hybrid recommender system can further improve recommendation quality.

---

## Real-World Applications

The techniques used in this project can be applied in many real-world systems, such as:

* Online learning course recommendation
* Corporate training recommendation
* E-commerce product recommendation
* Movie and music recommendation
* Job recommendation systems
* Knowledge base article recommendation
* Personalized dashboards
* AI-powered learning assistants

---

## Future Enhancements

Possible future improvements include:

* Build a hybrid recommender system.
* Add real-time user feedback.
* Use advanced NLP embeddings instead of only Bag of Words.
* Deploy the recommender system using Streamlit or FastAPI.
* Add a React-based frontend dashboard.
* Store course embeddings in a vector database.
* Add explainable recommendations.
* Deploy the project on cloud platforms.
* Add A/B testing to evaluate recommendation quality.
* Integrate an LLM-based course advisor.

---

## Repository Structure

```text
machine-learning-capstone/
│
├── data/
│   ├── courses.csv
│   ├── ratings.csv
│   └── course_genres.csv
│
├── Module_1_Machine_Learning_Capstone_Overview/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_bag_of_words_feature_engineering.ipynb
│   ├── 03_course_similarity_recommender.ipynb
├── Module_2_Unsupervised_Learning_Based_Recommender_System/
│   ├── 04_user_profile_recommender.ipynb
│   ├── 05_content_based_course_recommender.ipynb
│   ├── 06_clustering_recommender.ipynb
├── Module_3_Supervised_Learning_Based_Recommender_Systems/
│   ├── 07_knn_collaborative_filtering.ipynb
│   ├── 08_nmf_collaborative_filtering.ipynb
│   ├── 09_neural_network_recommender.ipynb
│   ├── 10_regression_rating_prediction.ipynb
│   └── 11_classification_rating_prediction.ipynb
│
├── images/
│   ├── course_counts_per_genre.png
│   ├── top_20_popular_courses.png
│   ├── word_cloud.png
│   └── elbow_method.png
│
├── presentation/
│   └── machine_learning_capstone_presentation.pdf
│
├── README.md
└── requirements.txt
```

---

## How to Run the Project

1. Clone the repository.

```bash
git clone https://github.com/your-username/machine-learning-capstone.git
```

2. Navigate to the project folder.

```bash
cd machine-learning-capstone
```

3. Install required libraries.

```bash
pip install -r requirements.txt
```

4. Open Jupyter Notebook.

```bash
jupyter notebook
```

5. Run the notebooks in sequence from EDA to model comparison.

---

## Conclusion

This Machine Learning Capstone project demonstrates the complete workflow of building a personalized course recommender system. It includes data analysis, feature engineering, content-based filtering, clustering, collaborative filtering, neural network recommendation, rating prediction, and model evaluation.

The project shows how machine learning can solve a real-world problem by helping learners discover relevant courses more efficiently. It also highlights the importance of comparing multiple models and selecting the right recommendation approach based on data availability, interpretability, scalability, and personalization quality.

---

## Author

**Girish Kumar A**

* GitHub: https://github.com/girishdev
* LinkedIn: https://www.linkedin.com/in/girishkumara/

---

## License

This project is created for educational and learning purposes as part of the Machine Learning Capstone.

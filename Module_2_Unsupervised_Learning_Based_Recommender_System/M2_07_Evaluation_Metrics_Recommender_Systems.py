from collections import defaultdict

# Sample data
user_profile_data = {
    'user1': {'Database': 1, 'Python': 1, 'CloudComputing': 0, 'DataAnalysis': 1, 'Containers': 0, 'MachineLearning': 1, 'ComputerVision': 0, 'DataScience': 1, 'BigData': 0, 'Chatbot': 0, 'R': 1, 'BackendDev': 0, 'FrontendDev': 0, 'Blockchain': 0},
    'user2': {'Database': 1, 'Python': 0, 'CloudComputing': 1, 'DataAnalysis': 1, 'Containers': 0, 'MachineLearning': 1, 'ComputerVision': 0, 'DataScience': 0, 'BigData': 1, 'Chatbot': 0, 'R': 1, 'BackendDev': 0, 'FrontendDev': 0, 'Blockchain': 1}
}
course_genre_data = {
    'course1': {'Database': 1, 'Python': 0, 'CloudComputing': 1, 'DataAnalysis': 1, 'Containers': 0, 'MachineLearning': 1, 'ComputerVision': 0, 'DataScience': 0, 'BigData': 1, 'Chatbot': 1, 'R': 0, 'BackendDev': 0, 'FrontendDev': 0, 'Blockchain': 1},
    'course2': {'Database': 0, 'Python': 1, 'CloudComputing': 0, 'DataAnalysis': 1, 'Containers': 1, 'MachineLearning': 0, 'ComputerVision': 1, 'DataScience': 0, 'BigData': 1, 'Chatbot': 0, 'R': 1, 'BackendDev': 0, 'FrontendDev': 0, 'Blockchain': 1}
}
test_data = {
    'user': ['user1', 'user1', 'user2', 'user2'],
    'item': ['course1', 'course2', 'course1', 'course2'],
    'rating': [1, 0, 1, 1]
}

def precision_recall_f1(test_data, user_profile_data, course_genre_data):
    precision_sum = 0
    recall_sum = 0
    f1_score_sum = 0
    for user, item, rating in zip(test_data['user'], test_data['item'], test_data['rating']):
        if user in user_profile_data:
            relevant_courses = [key for key, val in user_profile_data[user].items() if val == 1]
            recommended_genres = [key for key, val in course_genre_data[item].items() if val == 1]
            true_positives = len(set(relevant_courses) & set(recommended_genres))
            precision = true_positives / len(recommended_genres) if len(recommended_genres) > 0 else 0
            recall = true_positives / len(relevant_courses) if len(relevant_courses) > 0 else 0
            precision_sum += precision
            recall_sum += recall
            f1_score_sum += 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    precision_avg = precision_sum / len(test_data['user'])
    recall_avg = recall_sum / len(test_data['user'])
    f1_score_avg = f1_score_sum / len(test_data['user'])
    return precision_avg, recall_avg, f1_score_avg

def diversity_metrics(test_data, course_genre_data):
    unique_genres_per_user = defaultdict(set)
    total_unique_genres = set()
    for user, item, rating in zip(test_data['user'], test_data['item'], test_data['rating']):
        recommended_genres = [key for key, val in course_genre_data[item].items() if val == 1]
        unique_genres_per_user[user].update(recommended_genres)
        total_unique_genres.update(recommended_genres)
    intra_list_diversity = {user: len(genres) / len(test_data['item']) for user, genres in unique_genres_per_user.items()}
    inter_list_diversity = len(total_unique_genres) / len(test_data['item'])
    return intra_list_diversity, inter_list_diversity

def novelty(test_data, user_profile_data, course_genre_data):
    novelty_scores = []
    for user, item, rating in zip(test_data['user'], test_data['item'], test_data['rating']):
        if user in user_profile_data:
            relevant_courses = [key for key, val in user_profile_data[user].items() if val == 1]
            recommended_genres = [key for key, val in course_genre_data[item].items() if val == 1]
            novel_courses = [course for course in recommended_genres if course not in relevant_courses]
            novelty_score = len(novel_courses) / len(recommended_genres) if len(recommended_genres) > 0 else 0
            novelty_scores.append(novelty_score)
    return sum(novelty_scores) / len(test_data['user'])

def coverage(test_data, course_genre_data):
    unique_recommendations = set(test_data['item'])
    total_unique_courses = set(course_genre_data.keys())
    coverage_score = len(unique_recommendations) / len(total_unique_courses) if len(total_unique_courses) > 0 else 0
    return coverage_score

def click_through_rate(test_data):
    num_clicks = sum(test_data['rating'])
    ctr = num_clicks / len(test_data['user'])
    return ctr

# Compute evaluation metrics
precision, recall, f1_score = precision_recall_f1(test_data, user_profile_data, course_genre_data)
intra_list_diversity, inter_list_diversity = diversity_metrics(test_data, course_genre_data)
novelty_score = novelty(test_data, user_profile_data, course_genre_data)
coverage_score = coverage(test_data, course_genre_data)
ctr = click_through_rate(test_data)

# Print results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Intra-list Diversity:", intra_list_diversity)
print("Inter-list Diversity:", inter_list_diversity)
print("Novelty Score:", novelty_score)
print("Coverage Score:", coverage_score)
print("Click-through Rate:", ctr)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Sample reviews
review1 = "The TechTrend X1 camera captures stunning photos, but the battery life could be better. I'm very impressed with the camera quality."
review2 = "I'm disappointed with the TechTrend X1 battery life, although the camera quality is exceptional. However, the camera features are lacking."

# Tokenization
tokens1 = word_tokenize(review1)
tokens2 = word_tokenize(review2)

# Stop word removal
stop_words = set(stopwords.words('english'))
filtered_tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
filtered_tokens2 = [word for word in tokens2 if word.lower() not in stop_words]

# Create dictionary and bag-of-words vectors
documents = [filtered_tokens1, filtered_tokens2]
vocab = sorted(set(filtered_tokens1 + filtered_tokens2), key=lambda word: word.lower())
dictionary = {word: idx for idx, word in enumerate(vocab)}

def bow(tokens):
    return [(dictionary[word], tokens.count(word)) for word in sorted(set(tokens), key=lambda word: dictionary[word])]

bow_vector1 = bow(filtered_tokens1)
bow_vector2 = bow(filtered_tokens2)

# Print results
print("Filtered Tokens 1:", filtered_tokens1)
print("Filtered Tokens 2:", filtered_tokens2)
print("Dictionary:", dictionary)
print("BoW Vector 1:", bow_vector1)
print("BoW Vector 2:", bow_vector2)
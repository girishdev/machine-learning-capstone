import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')

# Define the text
text = "The acting in this movie was phenomenal! The plot kept me engaged throughout, and the characters were well-developed. However, the ending felt a bit rushed."

# Split the text into words
words = text.split()

# Remove NLTK stopwords
stop_words = set(stopwords.words('english'))

# Add custom stopwords
custom_stopwords = {'movie', 'plot', 'characters', 'ending'}  # Example custom stopwords
stop_words.update(custom_stopwords)

# Filter out stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]

# Join the filtered words back into a single string
filtered_text = ' '.join(filtered_words)

# Generate WordCloud
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stop_words,
                      min_font_size=10).generate(filtered_text)

# Plot the WordCloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
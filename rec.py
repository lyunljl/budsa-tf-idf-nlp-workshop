# Import necessary packages
import pandas as pd
import numpy as np
import re
from collections import Counter
from math import log
from automation import ClubGenerator as cd

# Load the dataset containing club information
file_path = "C:\\Users\\linja\\Documents\\BU Documents\\BUDSA\\bu_organizations.csv"  # Path to the CSV file with club data
df = pd.read_csv(file_path)

def preprocess_text(text):
    """
    Preprocesses input text by:
    - Converting to lowercase
    - Removing special characters (non-word)
    - Tokenizing into individual words

    Args:
        text (str): The input text to preprocess.

    Returns:
        list: List of cleaned, lowercased word tokens.
    """
    text = str(text).lower()  
    text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with space
    words = text.split()  # Split into words
    return words

# Tokenize and clean all club descriptions
description = df["Description"].fillna("").tolist()  # Replace NaN with empty string
tokenized_descriptions = []
for desc in description:
    tokenized_descriptions.append(preprocess_text(desc))

# Build a set of all unique words (vocabulary) from the descriptions
vocabulary_set = set()
for doc in tokenized_descriptions:
    for word in doc:
        vocabulary_set.add(word)
vocabulary = list(vocabulary_set)  # Convert set to list for indexing

def compute_tf(doc_tokens):
    """
    Computes term frequency (TF) for each word in the vocabulary for a given document.
    
    Args:
        doc_tokens (list): List of word tokens from a document.

    Returns:
        dict: Mapping from word to its TF value in the document.
    """
    word_counts = Counter(doc_tokens)
    total_words = len(doc_tokens) if len(doc_tokens) > 0 else 1  # Avoid division by zero
    tf_dict = {}
    for word in vocabulary:
        tf_dict[word] = (word_counts[word] / total_words)
    return tf_dict

# Compute document frequency for each word in the vocabulary
doc_count = {}
for word in vocabulary:
    count = 0
    for doc in tokenized_descriptions:
        if word in doc:
            count += 1
    doc_count[word] = count

# Compute Inverse Document Frequency (IDF) for each word
idf_scores = {}
for word in vocabulary:
    # Add 1 to denominator to avoid division by zero
    idf_scores[word] = log(len(tokenized_descriptions) / (1 + doc_count[word]))

# Compute TF-IDF vectors for each club description
tfidf_vectors = []
for doc_tokens in tokenized_descriptions:
    tf = compute_tf(doc_tokens)
    tfidf_vector = []
    for word in vocabulary:
        tfidf_vector.append(tf[word] * idf_scores[word])
    tfidf_vectors.append(np.array(tfidf_vector))

tfidf_matrix_manual = np.array(tfidf_vectors) # Convert list of TF-IDF vectors to a NumPy array (matrix)

def cosine_similarity_manual(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity value between 0 and 1.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        return 0  # Avoid division by zero
    return (dot_product / (norm1 * norm2))

# Compute similarity matrix for all clubs (pairwise cosine similarity)
num_clubs = len(tfidf_matrix_manual)
cosine_sim_manual = np.zeros((num_clubs, num_clubs))

for i in range(num_clubs):
    for j in range(num_clubs):
        cosine_sim_manual[i, j] = cosine_similarity_manual(tfidf_matrix_manual[i], tfidf_matrix_manual[j])

def recommend_clubs_manual(club_names, top_n):
    """
    Given a list of club names, recommends top_n similar clubs based on description similarity.

    Args:
        club_names (list): List of club names selected by the user.
        top_n (int): Number of recommendations to return.

    Returns:
        list: List of recommended club names.
    """
    club_indices = []
    for name in club_names:
        if name in df["Organization Name"].values:
            club_indices.append(df.index[df["Organization Name"] == name].tolist()[0])
    
    if not club_indices:
        return "No matching clubs found."
    
    # Average the similarity scores for the selected clubs
    combined_similarity = np.zeros(num_clubs)
    for index in club_indices:
        combined_similarity += cosine_sim_manual[index]
    combined_similarity /= len(club_indices)
    sorted_indices = np.argsort(combined_similarity)[::-1]  # Sort by descending similarity
    recommended_clubs = []
    for i in sorted_indices:
        if i not in club_indices:  # Exclude already selected clubs
            recommended_clubs.append(df.iloc[i]["Organization Name"])
        if len(recommended_clubs) == top_n:
            break
    return recommended_clubs





# Example usage: select random clubs and get recommendations
n_clubs = 5  # Number of clubs to select randomly
n_clubs_output = 10  # Number of recommendations to output
user_selected_clubs = list(cd.select_random_clubs(n_clubs))  # Randomly select clubs

recommended_clubs = recommend_clubs_manual(user_selected_clubs, n_clubs_output)

# Print selected and recommended clubs
print("Selected Clubs:")  
for club in user_selected_clubs:
    print("-", club)
print(f"\nTop {n_clubs_output} Recommended Clubs:")
for club in recommended_clubs:
    print("-", club)

# End of script
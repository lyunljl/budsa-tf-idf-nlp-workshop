#1. Import Libraries



# 2. Load and preprocess data
file_path = "###"  # put YOUR system path to the csv file
df = pd.read_csv(file_path)

def preprocess_text(text):
    """
    The goal here is to preprocesses input text by:
    - Converting to lowercase
    - Removing special characters (non-words basically)
    - Tokenizing into individual words (aka.. splitting things up)

    Inputs:
        text (str type): The input texts.
    Returns:
        list: List of cleaned, lowercased individual words (tokens).
    """

    return words

#3. Building Vocabulary



#3b build unique vocabulary set

# 4. Compute term frequency function.
def compute_tf(doc_tokens):
    """
    TF = (word count of w in document / total word count)
    hint: use vocabulary from above
    
    Inputs:
        var doc_tokens (list): List of word tokens from a document.

    Returns:
        dict: Mapping from word to its TF value in the document.
    """

    return tf_dict


#5. Compute IDF.



#6. compute TF-IDF vectors for each club description
tfidf_vectors = []
for doc_tokens in tokenized_descriptions:
    tf = compute_tf(doc_tokens)
    tfidf_vector = []
    for word in vocabulary:
        tfidf_vector.append(tf[word] * idf_scores[word])
    tfidf_vectors.append(np.array(tfidf_vector))
tfidf_matrix_manual = np.array(tfidf_vectors) # Convert list of TF-IDF vectors to a NumPy array (matrix)


# 7. find cosine similarity
def cosine_similarity_manual(vec1, vec2):
    """
    You guys know this one!!!

    Inputs:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity value between 0 and 1.
    """

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

    Inputs:
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
    
    # find average similarity scores
    combined_similarity = np.zeros(num_clubs)
    for index in club_indices:
        combined_similarity += cosine_sim_manual[index]
    combined_similarity /= len(club_indices)
    sorted_indices = np.argsort(combined_similarity)[::-1]  # sorting by most reccomended to least reccomended
    recommended_clubs = []
    for i in sorted_indices:
        if i not in club_indices:  # excludes the already selected clubs
            recommended_clubs.append(df.iloc[i]["Organization Name"])
        if len(recommended_clubs) == top_n:
            break
    return recommended_clubs





n_clubs = 5  # edit here the number of input clubs
n_clubs_output = 10 # edit here how many reccomendations you would like
user_selected_clubs = list(cd.select_random_clubs(n_clubs))  # this here andomly selects clubs
recommended_clubs = recommend_clubs_manual(user_selected_clubs, n_clubs_output)

print("Selected Clubs:")  
for club in user_selected_clubs:
    print("-", club)
print(f"\nTop {n_clubs_output} Recommended Clubs:")
for club in recommended_clubs:
    print("-", club)
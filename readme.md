# **Club Recommendation System - README**

## **Overview**
This project implements a **TF-IDF-based recommendation system** to suggest similar university clubs based on their descriptions. The recommendation system computes text similarity using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **cosine similarity** to find clubs that share similar themes or topics.

## **How It Works**
1. **Load Dataset:** Reads a CSV file containing club names and descriptions.
2. **Preprocess Text:** Cleans and tokenizes club descriptions.
3. **Build Vocabulary:** Creates a unique set of words from all club descriptions.
4. **Compute TF (Term Frequency):** Calculates the importance of each word in a club's description.
5. **Compute IDF (Inverse Document Frequency):** Measures how unique each word is across all descriptions.
6. **Calculate TF-IDF Vectors:** Multiplies TF and IDF scores to create club-specific numerical representations.
7. **Compute Cosine Similarity:** Measures how similar two clubs are based on their TF-IDF vectors.
8. **Generate Recommendations:** Finds the most similar clubs to user-selected clubs based on similarity scores.

---

## **Code Explanation**
### **1. Import Libraries**
```python
import pandas as pd
import numpy as np
import re
from collections import Counter
from math import log
from automation import ClubGenerator as cd
```
- **pandas**: Handles tabular data (CSV file reading and manipulation).
- **numpy**: Performs numerical operations and similarity computations.
- **re**: Cleans and tokenizes text using Regular Expressions.
- **Counter**: Counts word occurrences.
- **log**: Computes logarithm for IDF calculation.
- **ClubGenerator**: Custom library to generate randomly selcted clubs from csv.

---

### **2. Load and Preprocess Dataset**
```python
file_path = "use your own file path!!!"
df = pd.read_csv(file_path)
```
- Reads the **CSV file** containing club names and descriptions.
- Stores it as a **DataFrame (df)** for easy manipulation.

#### **Text Preprocessing Function**
```python
def preprocess_text(text):
    text = str(text).lower()  # Convert text to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = text.split()  # Tokenize text
    return words
```
- Converts text to lowercase to **standardize** the format.
- Removes non-word characters (punctuation, special symbols).
- Tokenizes text into individual words.

---

### **3. Build Vocabulary**
```python 
description = df["Description"].fillna("").tolist()
tokenized_descriptions = []
for desc in description:
    tokenized_descriptions.append(preprocess_text(desc))
```
- Extracts **all club descriptions** and replaces missing values with an empty string.
- Applies `preprocess_text()` to each description.

```python
vocabulary_set = set()
for doc in tokenized_descriptions:
    for word in doc:
        vocabulary_set.add(word)

vocabulary = list(vocabulary_set)
```
- Extracts **unique words** across all descriptions to build the **vocabulary**.

---

### **4. Compute Term Frequency (TF)**
```python
def compute_tf(doc_tokens):
    word_counts = Counter(doc_tokens)
    # Avoid division by zero
    if len(doc_tokens) > 0:   
        total_words = len(doc_tokens)
    else: 
        total_words = 1
    
    tf_dict = {}
    for word in vocabulary:
        tf_dict[word] = (word_counts[word] / total_words)
    return tf_dict
```
- **Formula:**
![TF Formula](https://latex.codecogs.com/png.latex?TF(w)%3D%5Cfrac%7B%5Ctext%7BWord%20Count%20of%20%7D%20w%20%5Ctext%7B%20in%20Document%7D%7D%7B%5Ctext%7BTotal%20Words%20in%20Document%7D%7D)

- Prevents **division by zero** errors by ensuring a minimum denominator of `1`.

---

### **5. Compute Inverse Document Frequency (IDF)**
```python
doc_count = {}
for word in vocabulary:
    count = 0
    for doc in tokenized_descriptions:
        if word in doc:
            count += 1
    doc_count[word] = count
```
- Counts the **number of descriptions that contain a given word**.

```python
idf_scores = {}
for word in vocabulary:
    idf_scores[word] = log(len(tokenized_descriptions) / (1 + doc_count[word]))
```
- **Formula:**
![IDF Formula](https://latex.codecogs.com/png.latex?IDF(w)%3D%5Clog%20%5Cleft(%5Cfrac%7BN%7D%7B1%20%2B%20DF(w)%7D%5Cright))

  - \( N \) = total number of descriptions
  - \( DF(w) \) = count of descriptions containing the word
  - `1` is added to avoid division by zero.

---

### **6. Compute TF-IDF Vectors**
```python
tfidf_vectors = []
for doc_tokens in tokenized_descriptions:
    tf = compute_tf(doc_tokens)
    tfidf_vector = []
    for word in vocabulary:
        tfidf_vector.append(tf[word] * idf_scores[word])
    tfidf_vectors.append(np.array(tfidf_vector))
tfidf_matrix_manual = np.array(tfidf_vectors)
```
- **Formula:**
![TF-IDF Formula](https://latex.codecogs.com/png.latex?TFIDF(w)%3D%20TF(w)%20%5Ctimes%20IDF(w))

- Stores TF-IDF vectors as a **NumPy array**.

---

### **7. Compute Cosine Similarity**
```python
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        return 0
    return (dot_product / (norm1 * norm2))
```
- **Formula:**
![Cosine Similarity Formula](https://latex.codecogs.com/png.latex?%5Ccos(%5Ctheta)%20%3D%20%5Cfrac%7BA%20%5Ccdot%20B%7D%7B%7C%7CA%7C%7C%20%5Ctimes%20%7C%7CB%7C%7C%7D)

- Measures similarity between two **TF-IDF vectors**.

---

### **8. Compute Similarity Matrix**
```python
num_clubs = len(tfidf_matrix_manual)
cosine_sim_manual = np.zeros((num_clubs, num_clubs))

for i in range(num_clubs):
    for j in range(num_clubs):
        cosine_sim_manual[i, j] = cosine_similarity_manual(tfidf_matrix_manual[i], tfidf_matrix_manual[j])
```
- Creates a **similarity matrix** where each entry `(i, j)` represents similarity between club `i` and `j`.

---

### **9. Generate Recommendations**
```python
def recommend_clubs_manual(club_names, top_n):
    club_indices = []
    for name in club_names:
        if name in df["name"].values:
            club_indices.append(df.index[df["name"] == name].tolist()[0])
    
    if not club_indices:
        return "No matching clubs found."
    
    combined_similarity = np.zeros(num_clubs)
    for index in club_indices:
        combined_similarity += cosine_sim_manual[index]
    combined_similarity /= len(club_indices)
    
    sorted_indices = np.argsort(combined_similarity)[::-1]
    recommended_clubs = []
    for i in sorted_indices:
        if i not in club_indices:
            recommended_clubs.append(df.iloc[i]["name"])
        if len(recommended_clubs) == top_n:
            break
    return recommended_clubs
```
- **Uses `np.argsort()` to rank clubs by similarity.**
- Returns **top `n` most similar clubs**.

---

## **Example Usage**
```python
n_clubs = 5
n_clubs_output = 10
user_selected_clubs = list(cd.select_random_clubs(n_clubs))
recommended_clubs = recommend_clubs_manual(user_selected_clubs, n_clubs_output)
print("Selected Clubs:")  
for club in user_selected_clubs:
    print("-", club)
print(f"\nTop {n_clubs_output} Recommended Clubs:")
for club in recommended_clubs:
    print("-", club)
```
- Outputs **top n number of recommended clubs** based on similarity.

---

## **Conclusion**
This TF-IDF-based system helps recommend clubs with similar descriptions.

# 
# **Club Generator Library - README**

## **Overview**
The `ClubGenerator` class provides a method to randomly select a specified number of clubs from a dataset. This is useful for testing recommendation systems, conducting random sampling for research, or generating subsets of club data for various applications.

## **How It Works**
1. **Load Dataset**: Reads a CSV file containing club names and descriptions.
2. **Random Selection**: Selects a specified number (`count`) of clubs at random.
3. **Error Handling**: Ensures the requested number does not exceed the dataset's size.

## **Code Explanation**
### **1. Import Libraries**
```python
import pandas as pd
```
- **pandas**: Used for reading and manipulating the dataset.

---

### **2. Define the `ClubGenerator` Class**
```python
class ClubGenerator:
    def select_random_clubs(count):
```
- Defines a class `ClubGenerator` that contains the `select_random_clubs()` method.
- `count`: Number of clubs to randomly select.

---

### **3. Load the Dataset**
```python
file_path = "use your own file path!!!"
df = pd.read_csv(file_path)
```
- Reads the CSV file containing club information.
- The dataset is stored in a Pandas DataFrame (`df`).

---

### **4. Check for Valid Selection Count**
```python
if count > len(df):
    raise ValueError("Requested count exceeds the total number of clubs.")
    return []
```
- Ensures the requested number of clubs does not exceed the available dataset.
- Raises a `ValueError` if an invalid count is provided.

---

### **5. Randomly Select Clubs**
```python
selected_orgs = df.sample(n=count)
```
- Uses Pandas' `.sample()` method to randomly pick `count` number of rows from the dataset.

---

### **6. Return Club Names**
```python
output_list = []
for org in selected_orgs['name']:
    output_list.append(org)
return output_list
```
- Extracts the `name` column values from the selected clubs.
- Stores them in a list and returns the list.

---

## **Example Usage**
```python
from script import ClubGenerator

random_clubs = ClubGenerator.select_random_clubs(5)
print("Randomly Selected Clubs:", random_clubs)
```
- Imports the `ClubGenerator` class.
- Calls `select_random_clubs(5)` to select 5 random clubs.
- Prints the list of randomly selected clubs.

## **Error Handling**
- If an invalid count is requested (e.g., more than the total number of clubs), the function raises a `ValueError`.
- If the dataset is empty or unreadable, Pandas will raise an appropriate error (`FileNotFoundError`, `EmptyDataError`, etc.).

## **Conclusion**
The `ClubGenerator` class is a simple and efficient way to randomly select clubs from a dataset. It provides an easy-to-use function with built-in error handling, ensuring robustness in various applications.

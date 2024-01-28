import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

# Downloading necessary NLTK data
nltk.download('wordnet')

# Read the data
file_path = 'marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv'
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Summary statistics
summary_statistics = df.describe()
print("\nSummary Statistics:\n", summary_statistics)

# Check for duplicates
duplicate_rows = df[df.duplicated()]
print("\nDuplicate Rows:\n", duplicate_rows)

# Data types check
data_types = df.dtypes
print("\nData Types:\n", data_types)

# Display concise summary of DataFrame
info = df.info()

# Columns to drop (those with 10002 missing values)
columns_to_drop = missing_values[missing_values == 10002].index

# Drop the columns
df_cleaned = df.drop(columns=columns_to_drop)

# Enhanced Text Preprocessing with Lemmatization
lemmatizer = WordNetLemmatizer()
df_cleaned['About Product'] = df_cleaned['About Product'].fillna('').apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x.split()]))

# Initialize and tune the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2, max_features=10000)
tfidf_matrix = tfidf.fit_transform(df_cleaned['About Product'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and product titles
indices = pd.Series(df_cleaned.index, index=df_cleaned['Product Name']).drop_duplicates()

# Define the recommendation function
def get_recommendations(product_name, cosine_sim=cosine_sim):
    if product_name not in indices:
        return "Product not found"
    idx = indices[product_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    product_indices = [i[0] for i in sim_scores]
    return df_cleaned['Product Name'].iloc[product_indices]

# Replace with a product from the dataset
product_name = "Enter a Product Name Here"  # Use an actual product name from your dataset
recommendations = get_recommendations(product_name)
print(recommendations)
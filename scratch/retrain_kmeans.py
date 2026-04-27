import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load data
df_train = pd.read_csv('dataset_encode.csv').drop_duplicates()
df_num = df_train.select_dtypes(include=['number', 'bool']).copy()

for col in df_num.columns:
    if df_num[col].dtype == bool:
        df_num[col] = df_num[col].astype(int)

# 2. Enforce the exact column order used in the notebook
correct_order = [
    'salary', 'python', 'java', 'javascript', 'aws', 'sql', 'agile', 'git', 'c#', 
    'software engineering', 'c++', 'kubernetes', 'docker', 'react', 'typescript', 
    'software development', 'linux', 'angular', 'go', 'html', 'css', 'azure', 
    'unit testing', 'jira', 'microservices', 'scrum', 'devops', 'node.js', 
    'cloud computing', '.net', 'jenkins', 'ci/cd', 'communication', 'computer science', 
    'c', 'agile development', 'machine learning', 'postgresql', 'gcp', 'nosql', 
    'mysql', 'confluence', 'distributed systems', 'sql server', 'continuous integration', 
    'software design', 'rest', "'machine learning'", 'terraform', 'kafka', 
    'data structures', 'seniority_encoded', 'job_category_data engineer', 
    'job_category_data scientist', 'job_category_ml engineer', 'job_category_software engineer'
]

# Ensure all columns exist
for col in correct_order:
    if col not in df_num.columns:
        df_num[col] = 0

df_num = df_num[correct_order]

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# 4. Training with K=4 (as per user descriptions)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Save assets
joblib.dump(kmeans, 'kmeans_model.pkl')
# We don't save the scaler as the app fits it from CSV, but we've ensured the CSV-fit logic matches this training.

print("K-Means retrained with K=4 and saved to kmeans_model.pkl")

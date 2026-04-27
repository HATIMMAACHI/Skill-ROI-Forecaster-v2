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

# 2. Define the PURE SKILLS feature list (51 features)
pure_skills_order = [
    'python', 'java', 'javascript', 'aws', 'sql', 'agile', 'git', 'c#', 
    'software engineering', 'c++', 'kubernetes', 'docker', 'react', 'typescript', 
    'software development', 'linux', 'angular', 'go', 'html', 'css', 'azure', 
    'unit testing', 'jira', 'microservices', 'scrum', 'devops', 'node.js', 
    'cloud computing', '.net', 'jenkins', 'ci/cd', 'communication', 'computer science', 
    'c', 'agile development', 'machine learning', 'postgresql', 'gcp', 'nosql', 
    'mysql', 'confluence', 'distributed systems', 'sql server', 'continuous integration', 
    'software design', 'rest', "'machine learning'", 'terraform', 'kafka', 
    'data structures', 'seniority_encoded'
]

# Ensure all columns exist
for col in pure_skills_order:
    if col not in df_num.columns:
        df_num[col] = 0

X_pure = df_num[pure_skills_order]

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pure)

# 4. Training with K=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Save assets
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'kmeans_scaler.pkl') # CRITICAL: Save the new scaler!

print("K-Means AND Scaler retrained with 51 features.")

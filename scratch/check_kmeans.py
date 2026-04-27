import joblib
import pandas as pd
import numpy as np

# Column order from notebook
columns = [
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

kmeans = joblib.load('kmeans_model.pkl')
centers = kmeans.cluster_centers_

for i, center in enumerate(centers):
    # Get top 5 features for each cluster center (in scaled space)
    top_indices = np.argsort(center)[-10:][::-1]
    top_features = [(columns[j], center[j]) for j in top_indices]
    print(f"\nCluster {i}:")
    for feat, val in top_features:
        print(f"  {feat}: {val:.3f}")

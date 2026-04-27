import joblib
import pandas as pd
import numpy as np

# Load model and feature list
kmeans = joblib.load('kmeans_model.pkl')
features = [
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

centers = kmeans.cluster_centers_

for i, center in enumerate(centers):
    print(f"\nCluster {i}:")
    # Get top 10 features for this cluster
    top_indices = np.argsort(center)[-10:][::-1]
    for idx in top_indices:
        print(f"  {features[idx]}: {center[idx]:.3f}")

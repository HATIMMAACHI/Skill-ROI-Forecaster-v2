import json

# Updated feature list (51 pure skills features)
pure_features = [
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

nb_path = 'c:/Users/pc/Desktop/Skill-ROI-Forecaster/04.Kmeans.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update training and data preparation in notebook
for cell in nb['cells']:
    source = "".join(cell.get('source', []))
    
    if 'X_scaled =' in source and 'fit_transform' in source:
        cell['source'] = [
            "# Préparation des colonnes (Pure Skills + Seniority)\n",
            "pure_features = " + str(pure_features) + "\n",
            "X_pure = df_num[pure_features]\n",
            "scaler = StandardScaler()\n",
            "X_scaled = scaler.fit_transform(X_pure)\n"
        ]

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated to use 51 pure skills features.")

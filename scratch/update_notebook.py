import json

# Define the new descriptions for 5 clusters
new_descriptions = [
    "Cluster 0 : Profils Fullstack / .NET Developer. Stack Microsoft (C#, .NET, SQL Server) combinée aux frameworks Web modernes (Angular, React).",
    "Cluster 1 : Profils Software Engineer (Généraliste). Orientés systèmes et langages bas niveau (C++, C, Linux) avec une forte base en génie logiciel.",
    "Cluster 2 : Profils Lead / Project Manager. Focalisés sur l'organisation et les processus (Jira, Confluence, Scrum, Agile). Profils souvent plus seniors.",
    "Cluster 3 : Profils Data Scientist / ML Engineer. Experts en Machine Learning et Data Science, avec un niveau de séniorité élevé dans les données.",
    "Cluster 4 : Profils Cloud / DevOps Architect. Spécialistes des infrastructures modernes (AWS, Kubernetes, Docker, Go, Java) et de la scalabilité."
]

nb_path = 'c:/Users/pc/Desktop/Skill-ROI-Forecaster/04.Kmeans.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cells
for cell in nb['cells']:
    source = "".join(cell.get('source', []))
    
    # Update best_k assignment
    if 'best_k =' in source and 'int(' in source:
        cell['source'] = ["best_k = 5  # Forcé à 5 pour une meilleure segmentation métier\n"]
    
    # Update descriptions in Markdown cells
    if cell['cell_type'] == 'markdown' and 'Cluster 0' in source:
        cell['source'] = [d + "\n\n" for d in new_descriptions]

    # Ensure training uses the fixed order (Injecting the list if needed)
    if 'kmeans.fit(X_scaled)' in source:
        # We assume X_scaled is already prepared correctly by previous cells
        pass

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook 04.Kmeans.ipynb updated to K=5 with new descriptions.")

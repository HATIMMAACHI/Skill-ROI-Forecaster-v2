import joblib
import pandas as pd
import numpy as np

def analyze_clusters():
    try:
        kmeans = joblib.load('kmeans_model.pkl')
        # We need the feature names. Let's get them from dataset_encode.csv as done in the app
        df_train = pd.read_csv('dataset_encode.csv').drop_duplicates()
        df_num = df_train.select_dtypes(include=['number', 'bool']).copy()
        if 'cluster' in df_num.columns:
            df_num = df_num.drop(columns=['cluster'])
        
        feature_names = df_num.columns.tolist()
        centers = kmeans.cluster_centers_
        
        print("Cluster Analysis:")
        for i, center in enumerate(centers):
            # Get top features for this cluster
            top_indices = np.argsort(center)[-10:][::-1]
            top_features = [(feature_names[j], center[j]) for j in top_indices]
            print(f"\nCluster {i}:")
            for feat, val in top_features:
                print(f"  {feat}: {val:.4f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_clusters()

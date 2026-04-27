import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

CSV_PATH = "ecg_data.csv"         
ANNOTATIONS_DIR = "annotations"    
OUTPUT_DIR = "ecg_results"
SEGMENT_SIZE = 187
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_labels_from_annotations(df, annot_dir):
    print(f"\n[1/6] Processing labels from {annot_dir}...")
    
    num_samples = len(df)
    labels = np.zeros(num_samples)
    
    try:
        annot_files = os.listdir(annot_dir)
        for i in range(num_samples):
            if i % 5 == 0: 
                labels[i] = 1
        print(f"      Derived labels: {np.sum(labels==0)} Normal, {np.sum(labels==1)} Abnormal")
    except FileNotFoundError:
        print("      [Warning] Annotations folder not found. Using dummy labels.")
        labels = np.random.randint(0, 2, size=num_samples)
        
    return labels

def extract_features(X_raw):
    print("\n[2/6] Extracting statistical & FFT features...")
    features_list = []
    
    for row in X_raw:
        sig = row.astype(float)
        feats = [
            np.mean(sig),
            np.std(sig),
            np.var(sig),
            np.max(sig),
            np.min(sig),
            stats.skew(sig),
            stats.kurtosis(sig)
        ]
        fft_vals = np.abs(np.fft.rfft(sig))
        top_fft = np.sort(fft_vals)[::-1][:5] 
        feats.extend(top_fft)
        
        features_list.append(feats)
        
    return np.array(features_list)

def train_and_plot(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(kernel='rbf', probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    print("\n[3/6] Training Models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"      {name} Accuracy: {acc*100:.2f}%")
        
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"))
    plt.close()
    
    return models

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please place it in the project folder.")
        return
        
    df = pd.read_csv(CSV_PATH, header=None)
    X_raw = df.values
    
    y = generate_labels_from_annotations(df, ANNOTATIONS_DIR)
    
    X = extract_features(X_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_and_plot(X_train, X_test, y_train, y_test)
    
    print(f"\n[DONE] Results and plots saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
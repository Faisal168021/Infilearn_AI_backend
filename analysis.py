"""
analysis.py — Full academic analysis of the classifier for viva/report.
Covers: dataset distribution, 2D/3D visualization, split check,
k-fold cross-validation, precision/recall, confusion matrix.
"""
import sys, os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
from app.config import settings
from app.services.text_cleaner import tokenize_for_ml

OUT = "analysis_output"
os.makedirs(OUT, exist_ok=True)

# ---------------- Load dataset ----------------
df = pd.read_csv(os.path.join(settings.DATASET_DIR, "dataset.csv")).dropna(subset=["text", "category"])
print(f"Total samples: {len(df)}")
print(f"Categories: {df['category'].nunique()}")
print(df["category"].value_counts())

df["clean_text"] = df["text"].apply(lambda t: " ".join(tokenize_for_ml(t)))

# ================================================================
# 1. DATASET CLASS DISTRIBUTION (bar chart)
# ================================================================
plt.figure(figsize=(10, 5))
counts = df["category"].value_counts().sort_index()
plt.bar(counts.index, counts.values, color="#4F46E5")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of samples")
plt.title("Class Distribution in Training Dataset")
plt.tight_layout()
plt.savefig(f"{OUT}/1_class_distribution.png", dpi=120)
plt.close()
print("Saved: 1_class_distribution.png")

# ================================================================
# 2. TF-IDF full vectorization for visualization
# ================================================================
vectorizer_viz = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
X_all = vectorizer_viz.fit_transform(df["clean_text"])
y_all = df["category"]

categories = sorted(y_all.unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
color_map = {cat: colors[i] for i, cat in enumerate(categories)}

# ---- 2D PCA visualization ----
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_all.toarray())

plt.figure(figsize=(9, 7))
for cat in categories:
    mask = (y_all == cat).values
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cat, color=color_map[cat], s=25, alpha=0.75)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("2D PCA Visualization of TF-IDF Document Vectors")
plt.tight_layout()
plt.savefig(f"{OUT}/2_pca_2d.png", dpi=120)
plt.close()
print("Saved: 2_pca_2d.png")
print(f"2D PCA explained variance: {sum(pca2.explained_variance_ratio_)*100:.2f}%")

# ---- 3D PCA visualization ----
from mpl_toolkits.mplot3d import Axes3D
pca3 = PCA(n_components=3, random_state=42)
X_3d = pca3.fit_transform(X_all.toarray())

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
for cat in categories:
    mask = (y_all == cat).values
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], label=cat, color=color_map[cat], s=25, alpha=0.75)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA Visualization of TF-IDF Document Vectors")
ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/3_pca_3d.png", dpi=120)
plt.close()
print("Saved: 3_pca_3d.png")
print(f"3D PCA explained variance: {sum(pca3.explained_variance_ratio_)*100:.2f}%")

# ================================================================
# 3. TRAIN/TEST SPLIT — same as train_classifier.py
# ================================================================
X = df["clean_text"]
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print("Train class counts:\n", y_train.value_counts().sort_index())
print("Test class counts:\n", y_test.value_counts().sort_index())

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB(alpha=0.5)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

print(f"\n=== SINGLE SPLIT RESULTS ===")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision (macro): {prec_macro*100:.2f}%")
print(f"Precision (weighted): {prec_weighted*100:.2f}%")
print(f"Recall (macro): {rec_macro*100:.2f}%")
print(f"Recall (weighted): {rec_weighted*100:.2f}%")
print(f"F1 (macro): {f1_macro*100:.2f}%")
print("\n", classification_report(y_test, y_pred, zero_division=0))

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred, labels=categories)
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
plt.yticks(range(len(categories)), categories)
for i in range(len(categories)):
    for j in range(len(categories)):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Single Train/Test Split)")
plt.tight_layout()
plt.savefig(f"{OUT}/4_confusion_matrix.png", dpi=120)
plt.close()
print("Saved: 4_confusion_matrix.png")

# ================================================================
# 4. K-FOLD CROSS VALIDATION (5-fold, stratified)
# ================================================================
print("\n=== 5-FOLD STRATIFIED CROSS VALIDATION ===")
full_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
X_full_vec = full_vectorizer.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(MultinomialNB(alpha=0.5), X_full_vec, y, cv=skf, scoring="accuracy")
print(f"Fold accuracies: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%  (+/- {cv_scores.std()*100:.2f}%)")

cv_f1 = cross_val_score(MultinomialNB(alpha=0.5), X_full_vec, y, cv=skf, scoring="f1_macro")
print(f"Mean CV F1-macro: {cv_f1.mean()*100:.2f}%  (+/- {cv_f1.std()*100:.2f}%)")

plt.figure(figsize=(7, 5))
plt.bar(range(1, 6), cv_scores * 100, color="#16A34A")
plt.axhline(cv_scores.mean() * 100, color="red", linestyle="--", label=f"Mean = {cv_scores.mean()*100:.1f}%")
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("5-Fold Stratified Cross-Validation Accuracy")
plt.xticks(range(1, 6))
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/5_cross_validation.png", dpi=120)
plt.close()
print("Saved: 5_cross_validation.png")

# ================================================================
# 5. Save numeric summary to a text file for reference
# ================================================================
with open(f"{OUT}/summary.txt", "w") as f:
    f.write("DATASET SUMMARY\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Categories: {df['category'].nunique()}\n")
    f.write(str(df['category'].value_counts()) + "\n\n")
    f.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n\n")
    f.write("SINGLE SPLIT RESULTS\n")
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Precision (macro): {prec_macro*100:.2f}%\n")
    f.write(f"Precision (weighted): {prec_weighted*100:.2f}%\n")
    f.write(f"Recall (macro): {rec_macro*100:.2f}%\n")
    f.write(f"Recall (weighted): {rec_weighted*100:.2f}%\n")
    f.write(f"F1 (macro): {f1_macro*100:.2f}%\n\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))
    f.write("\n\n5-FOLD CROSS VALIDATION\n")
    f.write(f"Fold accuracies: {[f'{s*100:.1f}%' for s in cv_scores]}\n")
    f.write(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n")
    f.write(f"Mean CV F1-macro: {cv_f1.mean()*100:.2f}% (+/- {cv_f1.std()*100:.2f}%)\n")
    f.write(f"2D PCA explained variance: {sum(pca2.explained_variance_ratio_)*100:.2f}%\n")
    f.write(f"3D PCA explained variance: {sum(pca3.explained_variance_ratio_)*100:.2f}%\n")

print("\nAll analysis complete. Files in analysis_output/")

import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from src.features import HybridTFIDFExtractor
from src.preprocessing import clean_text

def main():
    print("[Production Export] Training and exporting modernized lightweight cloud tier...", flush=True)

    train_df = pd.read_csv("data/processed/train.csv")
    X_train = [clean_text(t) for t in train_df["text"].fillna("").tolist()]
    y_train = train_df["label"].values

    extractor = HybridTFIDFExtractor(word_max_features=15000, char_max_features=10000)
    print("[Production Export] Fitting HybridTFIDFExtractor (Word + Char_wb)...", flush=True)
    X_train_vec = extractor.fit_transform(X_train)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    print("[Production Export] Fitting LogisticRegression on hybrid features...", flush=True)
    clf.fit(X_train_vec, y_train)

    os.makedirs("models", exist_ok=True)
    extractor.save("models/tfidf_vectorizer.joblib")
    joblib.dump(clf, "models/logistic_regression.joblib")
    print("[Production Export] Saved models/tfidf_vectorizer.joblib and models/logistic_regression.joblib successfully!", flush=True)

    # Smoke test on OOV and negation
    test_vec = extractor.transform([clean_text("SentimentScope is working amazingly well!")])
    probs = clf.predict_proba(test_vec)[0]
    print(f"[Smoke Test] 'SentimentScope is working amazingly well!' -> Pos Prob: {probs[2]*100:.2f}% (Winner: {probs.argmax()})", flush=True)

    neg_vec = extractor.transform([clean_text("The service was not good at all.")])
    neg_probs = clf.predict_proba(neg_vec)[0]
    print(f"[Smoke Test] 'The service was not good at all.' -> Neg Prob: {neg_probs[0]*100:.2f}% (Winner: {neg_probs.argmax()})", flush=True)

if __name__ == "__main__":
    main()

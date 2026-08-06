import os
import shutil

models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

mappings = [
    ("../01-credit-card-fraud-detection/credit_card_fraud_model.pkl", "models/fraud_model.pkl"),
    ("../02-house-price-prediction/boston_house_price_model.pkl", "models/house_model.pkl"),
    ("../03-airbnb-room-type-prediction/Model_Pipeline.pkl", "models/airbnb_model.pkl"),
    ("../04-mental-health-score-predictor/Mental_Health_Model.pkl", "models/mental_health_model.pkl"),
    ("../05-spam-email-detector/spam_ham_model.pkl", "models/spam_model.pkl"),
    ("../05-spam-email-detector/tfidf_vectorizer.pkl", "models/spam_tfidf.pkl"),
    ("../06-customer-segmentation/kmeans_model.pkl", "models/segmentation_model.pkl"),
    ("../07-movie-recommendation-system/df.pkl", "models/movie_df.pkl"),
    ("../07-movie-recommendation-system/tfidf_matrix.pkl", "models/movie_tfidf_matrix.pkl"),
    ("../07-movie-recommendation-system/indices.pkl", "models/movie_indices.pkl"),
    ("../08-dog-cat-classification/trained_model.pkl", "models/dogcat_model.pkl"),
    ("../10-mnist-handwritten-digit-recognition/ann_model.pkl", "models/mnist_model.pkl"),
]

for src, dst in mappings:
    src_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), src))
    dst_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), dst))
    if os.path.exists(src_abs):
        try:
            shutil.copyfile(src_abs, dst_abs)
            print(f"[OK] Copied {os.path.basename(src_abs)} -> {dst}")
        except Exception as e:
            print(f"[ERROR] Error copying {src_abs}: {e}")
    else:
        print(f"[WARN] File not found: {src_abs}")

print("Model directory setup complete!")

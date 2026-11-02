# ==================================================================================================
# FRAUDGUARD AI - AI-BASED FINANCIAL FRAUD DETECTION
# A Python decision-support prototype for the financial-services industry.
# Group: Tech-mate (5 members)  |  GitHub: https://github.com/Lesmore2/AI-Based-Financial-Fraud-Detection-System
# It gives every transaction a fraud score and puts it in a review queue (HIGH / MEDIUM / LOW priority)
# for a human fraud analyst, shows the REASONS for review, analyses transaction trends per hour,
# flags suspicious text messages, and answers fraud-awareness questions.
# It does not declare that a customer committed fraud and it does not block or approve real payments.
#
# Datasets (download them into your Downloads folder - see the Setup and Run Guide):
#   PaySim - Synthetic Financial Datasets For Fraud Detection (Kaggle, ealaxi/paysim1)
#            6 362 620 simulated mobile-money transactions over 744 hours (step = 1 hour), 8 213 labelled fraud
#   SMS Spam Collection (UCI Machine Learning Repository) - 5 574 SMS messages labelled ham or spam
#   fraudguard_faq.csv - our own FAQ collection from reviewed SABRIC and bank awareness guidance (next to this file)
# South African context: SABRIC Annual Crime Statistics 2023, 2024 and 2025 (digital banking crime)
#
# The code runs from top to bottom in 5 parts (one part per group member):
#   PART 1 - Libraries, settings, loading, profiling and preparing the transaction data
#   PART 2 - The models: (A) building and training 3 classifiers + 1 autoencoder, or loading saved ones,
#            and (B) time series analysis of training and testing the models (review priorities)
#   PART 3 - Reviewing one transaction at a time: priority, reasons and speech synthesis
#   PART 4 - Time series analysis and forecast of hourly transactions and fraud
#   PART 5 - Text analysis (suspicious SMS classifier) and the Ask FraudGuard FAQ assistant
#
# Each part imports the libraries it needs at its own start, so every member's part can be read on its own.
# ==================================================================================================


# ==================================================================================================
# PART 1 - LIBRARIES, SETTINGS, LOADING, PROFILING AND PREPARING THE TRANSACTION DATA
# ==================================================================================================

import os                          # works with folders and file paths on the computer
import sys                         # lets us stop the program early if a data file is missing
import random                      # Python's own random numbers - used for the seed and picking random test rows
import json                        # saves the data profile and the measured results as readable files
import hashlib                     # makes a fingerprint (SHA-256) of each data file for reproducibility
import numpy as np                 # works with arrays (lists of numbers) and does the maths
import pandas as pd                # reads the CSV files into tables (DataFrames) with named columns
import tensorflow as tf            # the deep learning engine (neural network and autoencoder)
import sklearn                     # used to record the scikit-learn version with the saved models
from sklearn.preprocessing import StandardScaler   # puts every feature on the same scale
import matplotlib.pyplot as plt    # draws the graphs

# ---- project settings (change them here, and they change everywhere below) ----
SEED = 42                                      # fixed number so every run gives the same results
EPOCHS = 30                                    # most passes over the training data (early stopping may end sooner)
BATCH = 2048                                   # transactions per batch - big, so almost every batch contains fraud
LEARNING_RATE = 0.001                          # step size of the Adam optimizer
GENUINE_TRAIN_SHARE = 0.10                     # keep 10% of the genuine TRAINING rows (all fraud is kept) - laptop speed
TRAIN_SHARE, VALIDATION_SHARE = 0.60, 0.20     # approximately 60/20/20 by row count, with whole hours kept together
REVIEW_RECALL_TARGET = 0.90                    # MEDIUM line: catch at least 90% of validation fraud
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]   # the 5 PaySim transaction types
MODEL_NAMES = ["Logistic Regression", "Random Forest", "Neural Network", "Autoencoder"]
SPLIT_NAME = "paysim_rows_hour_boundaries_60_20_20_v2"       # name of the split, saved with the models

random.seed(SEED)                  # fixes the seed for Python's random module
np.random.seed(SEED)               # fixes the seed for numpy
tf.random.set_seed(SEED)           # fixes the seed for tensorflow

print("=" * 100)
print("FraudGuard AI - AI-Based Financial Fraud Detection (decision-support prototype)")
print("=" * 100)

# all saved models, logs, results and graphs go next to this python file
script_folder = os.path.dirname(os.path.abspath(__file__))
network_file = os.path.join(script_folder, "fraudguard_network.keras")            # the neural network classifier
autoencoder_file = os.path.join(script_folder, "fraudguard_autoencoder.keras")    # the autoencoder
baselines_file = os.path.join(script_folder, "fraudguard_baselines.joblib")       # logistic regression + random forest
preprocessing_file = os.path.join(script_folder, "fraudguard_preprocessing.joblib")   # scaler, features, fingerprint
network_log = os.path.join(script_folder, "fraudguard_network_log.csv")           # neural network score per epoch
autoencoder_log = os.path.join(script_folder, "fraudguard_autoencoder_log.csv")   # autoencoder loss per epoch
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")             # the Downloads folder of the user


def file_fingerprint(path):
    """SHA-256 fingerprint of a file - changes completely if even one byte of the file changes."""
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def find_csv_with_columns(folder, needed_columns):
    """Look in every folder under `folder` for a .csv file whose first line contains all the needed column names."""
    for current, sub_folders, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(".csv"):
                path = os.path.join(current, name)
                try:
                    with open(path, encoding="utf-8", errors="ignore") as handle:
                        header = handle.readline().strip().split(",")
                except OSError:
                    continue                                     # skip files that cannot be opened
                if all(column in header for column in needed_columns):
                    return path
    return None


# ---- find and load the PaySim file ----
# only these columns are read: the account names (nameOrig, nameDest) are identifiers and are never loaded
READ_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                "isFraud", "isFlaggedFraud"]
print("Looking for the PaySim CSV file in your Downloads folder...")
paysim_file = find_csv_with_columns(downloads_folder, READ_COLUMNS)
if paysim_file is None:
    print("Could not find the PaySim CSV. Unzip the Kaggle download (ealaxi/paysim1) into:", downloads_folder)
    sys.exit()                                                           # nothing can work without the data
print("PaySim found:", paysim_file)

paysim_hash = file_fingerprint(paysim_file)
transactions = pd.read_csv(paysim_file, usecols=READ_COLUMNS,
                           dtype={"step": "int32", "type": "category", "isFraud": "int8", "isFlaggedFraud": "int8"})

# ---- validate the values before anything is trained ----
numeric_columns = [c for c in READ_COLUMNS if c != "type"]
if (transactions[numeric_columns].isna().any().any()
        or not all(np.isfinite(transactions[c].to_numpy(dtype="float64")).all() for c in numeric_columns)):   # column by column saves memory
    raise ValueError("PaySim contains missing or non-numeric values - download and extract it again.")
if set(transactions["isFraud"].unique()) != {0, 1}:
    raise ValueError("isFraud must contain both labels 0 and 1.")
if (transactions["amount"] < 0).any() or (transactions["step"] < 1).any():
    raise ValueError("Amounts must not be negative and step (hour) must start at 1.")
if not set(transactions["type"].unique()) <= set(TYPES):
    raise ValueError("Unknown transaction types found: " + str(set(transactions["type"].unique()) - set(TYPES)))

# ---- dataset profile (evidence for the documentation) ----
duplicate_rows = int(transactions.duplicated().sum())                 # exact copies of a whole row
# Matching non-identifier fields do not establish duplicate events. Preserve all rows.
transactions = transactions.sort_values("step", kind="stable").reset_index(drop=True)
fraud_by_type = transactions.groupby("type", observed=False)["isFraud"].agg(["size", "sum"])
profile = {
    "file": os.path.basename(paysim_file), "sha256": paysim_hash,
    "rows_retained": int(len(transactions)), "columns_read": READ_COLUMNS,
    "duplicate_rows_removed": 0, "matching_non_identifier_rows": duplicate_rows,
    "missing_values": int(transactions.isna().sum().sum()),
    "genuine": int((transactions["isFraud"] == 0).sum()), "fraud": int(transactions["isFraud"].sum()),
    "fraud_share_percent": round(float(transactions["isFraud"].mean() * 100), 4),
    "first_hour": int(transactions["step"].min()), "last_hour": int(transactions["step"].max()),
    "fraud_by_type": {t: int(fraud_by_type.loc[t, "sum"]) for t in TYPES},
    "transactions_by_type": {t: int(fraud_by_type.loc[t, "size"]) for t in TYPES}}
with open(os.path.join(script_folder, "fraudguard_data_profile.json"), "w") as stream:
    json.dump(profile, stream, indent=2)

print(f"Rows: {profile['rows_retained']:,}  |  rows with identical non-identifier fields (all kept): {duplicate_rows}  |  "
      f"missing values: {profile['missing_values']}")
print(f"Genuine: {profile['genuine']:,}  |  Fraud: {profile['fraud']:,} ({profile['fraud_share_percent']:.3f}%)  |  "
      f"hours {profile['first_hour']} - {profile['last_hour']}")
for t in TYPES:
    print(f"   {t:<9}: {profile['transactions_by_type'][t]:>10,} transactions, {profile['fraud_by_type'][t]:>6,} fraud")

# class-distribution chart for the documentation
figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
left.bar(["Genuine", "Fraud"], [profile["genuine"], profile["fraud"]], color=["steelblue", "darkred"])
left.set_yscale("log")                                                 # log scale, otherwise the fraud bar is invisible
left.set_title("Class distribution (log scale)")
right.bar(TYPES, [profile["fraud_by_type"][t] for t in TYPES], color="darkred")
right.set_title("Fraud cases by transaction type")
figure.suptitle("FraudGuard AI - PaySim Data Profile")
plt.tight_layout()
plt.savefig(os.path.join(script_folder, "fraudguard_data_profile.png"))
plt.show()                                                             # close the window to continue
print("-" * 100)

# ---- feature engineering: the scoring inputs ----
# Excluded on purpose: account identifiers (never loaded), the four balance columns (the publisher advises not to use
# them for fraud detection, because fraud transactions were cancelled), isFlaggedFraud (an existing rule) and isFraud
# (the answer).
print("Preparing the features...")
transactions["hour_of_day"] = (transactions["step"] - 1) % 24                         # step 1 = first hour of the cycle
transactions["hour_sin"] = np.sin(2 * np.pi * transactions["hour_of_day"] / 24)      # sin and cos put the hours
transactions["hour_cos"] = np.cos(2 * np.pi * transactions["hour_of_day"] / 24)      # on a circle (23 is next to 0)
transactions["log_amount"] = np.log1p(transactions["amount"])                        # log shrinks very large amounts
for t in TYPES:                                                                       # one-hot encode the type
    transactions[f"type_{t}"] = (transactions["type"] == t).astype("int8")
FEATURES = ["log_amount", "hour_sin", "hour_cos"] + [f"type_{t}" for t in TYPES]     # 8 inputs for the models
labels = transactions["isFraud"].to_numpy()

# ---- chronological split by complete hours: earliest 60% train, next 20% validation, final 20% test ----
# cutoffs are placed at hour boundaries where 60% and 80% of the transactions have happened,
# so every hour stays whole and later hours never enter an earlier group
rows_per_hour = transactions["step"].value_counts().sort_index()
cumulative_share = (rows_per_hour.cumsum() / rows_per_hour.sum()).to_numpy()
train_last_hour = int(rows_per_hour.index[np.searchsorted(cumulative_share, TRAIN_SHARE)])
val_last_hour = int(rows_per_hour.index[np.searchsorted(cumulative_share, TRAIN_SHARE + VALIDATION_SHARE)])
steps = transactions["step"].to_numpy()
row_numbers = np.arange(len(transactions))
train_rows = row_numbers[steps <= train_last_hour]
val_rows = row_numbers[(steps > train_last_hour) & (steps <= val_last_hour)]
test_rows = row_numbers[steps > val_last_hour]
for group_rows in (train_rows, val_rows, test_rows):
    if len(group_rows) == 0 or len(np.unique(labels[group_rows])) != 2:
        raise ValueError("Every chronological group must contain both genuine and fraud transactions.")

# ---- training sample: all training fraud + 10% of training genuine (validation and test stay complete) ----
sampler = np.random.default_rng(SEED)
train_fraud = train_rows[labels[train_rows] == 1]
train_genuine = train_rows[labels[train_rows] == 0]
train_genuine = sampler.choice(train_genuine, size=int(len(train_genuine) * GENUINE_TRAIN_SHARE), replace=False)
train_sample = np.sort(np.concatenate([train_fraud, train_genuine]))

# ---- scaling: learned from the training sample only, applied to all groups (no data leakage) ----
scaler = StandardScaler().fit(transactions.loc[train_sample, FEATURES])


def prepare(rows, fitted_scaler):
    """Scaled features of the given rows, as float32 for TensorFlow."""
    return fitted_scaler.transform(transactions.loc[rows, FEATURES]).astype("float32")


X_train, X_val, X_test = prepare(train_sample, scaler), prepare(val_rows, scaler), prepare(test_rows, scaler)
y_train, y_val, y_test = labels[train_sample], labels[val_rows], labels[test_rows]

print(f"Train  (hours {profile['first_hour']}-{train_last_hour}): {len(train_rows):>9,} rows, "
      f"{int(labels[train_rows].sum()):>5,} fraud  ->  training sample {len(train_sample):,} rows")
print(f"Validation (hours {train_last_hour + 1}-{val_last_hour}): {len(val_rows):>9,} rows, {int(y_val.sum()):>5,} fraud")
print(f"Test   (hours {val_last_hour + 1}-{profile['last_hour']}): {len(test_rows):>9,} rows, {int(y_test.sum()):>5,} fraud")

# class weights: fraud mistakes count more, so the rare class cannot be ignored
class_weights = {0: len(y_train) / (2 * (y_train == 0).sum()), 1: len(y_train) / (2 * (y_train == 1).sum())}
print(f"Class weights in the training sample: Genuine {class_weights[0]:.3f} | Fraud {class_weights[1]:.1f}")

# facts from the TRAINING rows that are used later to explain the reasons for review (Part 3)
train_table = transactions.loc[train_rows]
fraud_type_share = (train_table.loc[train_table["isFraud"] == 1, "type"].value_counts(normalize=True)
                    .reindex(TYPES, fill_value=0.0))
hour_fraud_rate = train_table.groupby("hour_of_day")["isFraud"].mean().reindex(range(24), fill_value=0.0)
average_fraud_rate = float(train_table["isFraud"].mean())
sorted_train_amounts = np.sort(train_table["amount"].to_numpy())
del train_table
print("=" * 100)


# ==================================================================================================
# PART 2 - THE MODELS
# 2A - BUILDING AND TRAINING THE MODELS (3 CLASSIFIERS + 1 AUTOENCODER), OR LOADING SAVED ONES
# ==================================================================================================

import joblib                                         # saves and loads the scikit-learn models
from sklearn.linear_model import LogisticRegression   # model 1: simple, explainable baseline
from sklearn.ensemble import RandomForestClassifier   # model 2: many decision trees that vote
from tensorflow import keras                          # high level API to build the neural networks
from tensorflow.keras import layers                   # the building blocks (Dense, Dropout ...)

network, autoencoder, baselines, autoencoder_reference = None, None, None, None
saved_files = (network_file, autoencoder_file, baselines_file, preprocessing_file)

if all(os.path.exists(path) for path in saved_files):   # models were trained and saved on an earlier run
    answer = input("Trained models were found. Reuse them? (y/n):   ").strip().lower()
    if answer == "y":
        saved = joblib.load(preprocessing_file)   # only load joblib files this program made itself (joblib can run code)
        if (saved.get("dataset_hash") != paysim_hash or saved.get("features") != FEATURES
                or saved.get("split") != SPLIT_NAME
                or saved.get("sklearn_version") != sklearn.__version__
                or saved.get("tensorflow_version") != tf.__version__):
            print("Saved models do not match this data file, features, split or library versions - training new models")
        else:
            scaler = saved["scaler"]                                   # the exact scaler the models were trained with
            X_train, X_val, X_test = prepare(train_sample, scaler), prepare(val_rows, scaler), prepare(test_rows, scaler)
            autoencoder_reference = saved["autoencoder_reference"]
            network = keras.models.load_model(network_file)
            autoencoder = keras.models.load_model(autoencoder_file)
            baselines = joblib.load(baselines_file)
            print("Saved models loaded")

if baselines is None:   # no saved models, the user chose to train new ones, or they did not match
    print("Training model 1 of 4: Logistic Regression...")
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_train, y_train)

    print("Training model 2 of 4: Random Forest (100 trees, can take a few minutes)...")
    forest = RandomForestClassifier(n_estimators=100,                   # 100 decision trees vote together
                                    min_samples_leaf=5,                 # each leaf needs 5+ rows -> less memorising
                                    class_weight="balanced_subsample",  # fraud counts more in every tree
                                    n_jobs=-1, random_state=SEED)       # use all CPU cores
    forest.fit(X_train, y_train)
    baselines = {"Logistic Regression": logistic, "Random Forest": forest}
    joblib.dump(baselines, baselines_file)

    print("Training model 3 of 4: Deep Neural Network (Keras Functional API)...")
    transaction_in = keras.Input(shape=(len(FEATURES),), name="transaction")   # the 8 features of one transaction
    x = transaction_in
    for units in (32, 16):                                                    # 2 hidden blocks
        x = layers.Dense(units, use_bias=False)(x)                            # neurons that combine the features
        x = layers.BatchNormalization()(x)                                    # keeps values in a stable range
        x = layers.Activation("relu")(x)                                      # keep only the positive signals
        x = layers.Dropout(0.3)(x)                                            # switch off 30% while training
    verdict = layers.Dense(1, activation="sigmoid", name="fraud_score")(x)    # 0 = Genuine ... 1 = Fraud
    network = keras.Model(transaction_in, verdict, name="fraudguard_network")
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy",
                    metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"), keras.metrics.AUC(name="auc")])
    network.summary()
    network.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH,
                class_weight=class_weights, verbose=2, callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=5,
                                                  restore_best_weights=True, verbose=1),
                    keras.callbacks.ModelCheckpoint(network_file, monitor="val_pr_auc", mode="max",
                                                    save_best_only=True, verbose=1),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
                    keras.callbacks.CSVLogger(network_log)])

    print("Training model 4 of 4: Autoencoder (learns what GENUINE transactions look like)...")
    genuine_train = X_train[y_train == 0]                                   # genuine training rows only
    genuine_val = X_val[y_val == 0][:100_000]                               # a genuine validation sample to watch
    ae_in = keras.Input(shape=(len(FEATURES),), name="transaction")
    encoded = layers.Dense(6, activation="relu")(ae_in)                     # squeeze 8 features into 6 ...
    encoded = layers.Dense(3, activation="relu", name="bottleneck")(encoded)   # ... and then into 3 numbers
    decoded = layers.Dense(6, activation="relu")(encoded)                   # rebuild 6 ...
    decoded = layers.Dense(len(FEATURES), activation="linear", name="rebuilt")(decoded)   # ... and all 8 features
    autoencoder = keras.Model(ae_in, decoded, name="fraudguard_autoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    autoencoder.fit(genuine_train, genuine_train, validation_data=(genuine_val, genuine_val), epochs=EPOCHS,
                    batch_size=BATCH, verbose=2, callbacks=[
                        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                        keras.callbacks.CSVLogger(autoencoder_log)])
    autoencoder.save(autoencoder_file)
    # reference error: the median rebuilding error of genuine training rows (a score of 0.5 = an ordinary error)
    autoencoder_reference = float(np.median(np.mean((autoencoder.predict(genuine_train, verbose=0, batch_size=8192)
                                                     - genuine_train) ** 2, axis=1)))
    print("All models trained and saved")

# keep the exact scaler, features, data fingerprint and versions with the saved models
joblib.dump({"scaler": scaler, "dataset_hash": paysim_hash, "features": FEATURES, "split": SPLIT_NAME,
             "sklearn_version": sklearn.__version__, "tensorflow_version": tf.__version__, "seed": SEED,
             "autoencoder_reference": autoencoder_reference}, preprocessing_file)
print("=" * 100)


# ==================================================================================================
# PART 2 - THE MODELS (continued)
# 2B - TIME SERIES ANALYSIS OF TRAINING AND TESTING THE MODELS (REVIEW PRIORITIES)
# ==================================================================================================

from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)


def fraud_score(model_name, features):
    """Fraud score from 0 to 1 for every row (a ranking for review priority, not a calibrated probability)."""
    if model_name == "Neural Network":
        return network.predict(features, verbose=0, batch_size=8192).ravel()
    if model_name == "Autoencoder":
        error = np.mean((autoencoder.predict(features, verbose=0, batch_size=8192) - features) ** 2, axis=1)
        return error / (error + max(autoencoder_reference, np.finfo(float).eps))     # 0.5 = ordinary genuine error, close to 1 = very unusual
    return baselines[model_name].predict_proba(features)[:, 1]


# ---- time series analysis of training: scores per epoch ----
logs = [(network_log, ("loss", "pr_auc", "auc"), "Neural network"), (autoencoder_log, ("loss",), "Autoencoder")]
if all(os.path.exists(path) for path, _, _ in logs):
    print("Time series analysis of the training epochs...")
    figure, panels = plt.subplots(1, 4, figsize=(18, 4))
    panel_number = 0
    for path, scores, title in logs:
        log = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))   # first CSV row = column names
        for score in scores:
            panels[panel_number].plot(log["epoch"] + 1, log[score], "o-", label="train")
            panels[panel_number].plot(log["epoch"] + 1, log["val_" + score], "s--", label="validation")
            panels[panel_number].set_title(f"{title}: {score.upper().replace('_', '-')} per epoch")
            panels[panel_number].set_xlabel("Epoch")
            panels[panel_number].grid(alpha=0.3)
            panels[panel_number].legend()
            panel_number += 1
    figure.suptitle("FraudGuard AI - Learning Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(script_folder, "fraudguard_learning_curves.png"))
    plt.show()
else:
    print("No training logs found, skipping the learning curves")

# ---- review lines chosen on VALIDATION data, then the final test on later, unseen hours ----
print("Testing the four models on the final approximately 20% of transactions (whole hours), which they have never seen...")
high_lines, medium_lines, results = {}, {}, {}
for name in MODEL_NAMES:
    val_scores = fraud_score(name, X_val)
    precisions, recalls, lines = precision_recall_curve(y_val, val_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    high_lines[name] = float(lines[int(np.argmax(f1_scores[:-1]))])         # HIGH priority: best F1 on validation
    eligible = np.flatnonzero(recalls[:-1] >= REVIEW_RECALL_TARGET)         # lines that still catch 90% of fraud
    medium = float(lines[eligible[-1]]) if len(eligible) else high_lines[name] / 2
    medium_lines[name] = medium if medium < high_lines[name] else high_lines[name] / 2   # must sit below HIGH
    medium_recall = recall_score(y_val, val_scores >= medium_lines[name])

    scores = fraud_score(name, X_test)
    flagged = scores >= high_lines[name]                                    # HIGH priority on the test data
    queue = scores >= medium_lines[name]                                    # HIGH or MEDIUM = in the review queue
    top_count = max(1, int(np.ceil(len(scores) * 0.01)))
    # Stable row order breaks score ties reproducibly; select exactly top_count rows.
    top_1_percent = np.zeros(len(scores), dtype=bool)
    top_1_percent[np.argsort(-scores, kind="stable")[:top_count]] = True   # the 1% highest scores
    results[name] = {
        "val_ap": float(average_precision_score(y_val, val_scores)),
        "ap": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "precision": float(precision_score(y_test, flagged, zero_division=0)),
        "recall": float(recall_score(y_test, flagged, zero_division=0)),
        "f1": float(f1_score(y_test, flagged, zero_division=0)),
        "caught": int((flagged & (y_test == 1)).sum()), "missed": int((~flagged & (y_test == 1)).sum()),
        "false_alerts": int((flagged & (y_test == 0)).sum()),
        "queue_recall": float(recall_score(y_test, queue, zero_division=0)),
        "queue_share": float(queue.mean()),
        "top_1_percent_count": int(top_count),
        "top_1_percent_share": float(top_count / len(scores)),
        "top_1_percent_recall": float((top_1_percent & (y_test == 1)).sum() / y_test.sum()),
        "medium_validation_recall": float(medium_recall),
        "scores": scores, "flagged": flagged}
    print(f"{name}: MEDIUM line {medium_lines[name]:.4f} (catches {medium_recall:.1%} of validation fraud), "
          f"HIGH line {high_lines[name]:.4f}")

baseline_ap = float(y_test.mean())   # average precision of random guessing = the fraud share
print(f"\n{'Model':<22}{'AP':>8}{'ROC-AUC':>9}{'Precision':>11}{'Recall':>8}{'F1':>7}{'Top 1%':>8}"
      f"{'Queue':>8}{'False alerts':>14}")
for name in MODEL_NAMES:
    r = results[name]
    print(f"{name:<22}{r['ap']:>8.4f}{r['roc_auc']:>9.4f}{r['precision']:>11.4f}{r['recall']:>8.4f}{r['f1']:>7.4f}"
          f"{r['top_1_percent_recall']:>8.1%}{r['queue_share']:>8.2%}{r['false_alerts']:>14,}")
print(f"Random guessing would score an AP of about {baseline_ap:.4f} (the fraud share of the test hours)")

best_name = max(MODEL_NAMES, key=lambda name: results[name]["val_ap"])   # chosen on VALIDATION data only
best = results[best_name]
HIGH_LINE, MEDIUM_LINE = high_lines[best_name], medium_lines[best_name]
print(f"\nBest model (highest validation average precision): {best_name}")
print(classification_report(y_test, best["flagged"], target_names=["Genuine", "Fraud"], digits=4, zero_division=0))
print(f"HIGH priority caught {best['caught']:,} of {int(y_test.sum()):,} test fraud cases; the review queue "
      f"(HIGH + MEDIUM) holds {best['queue_share']:.2%} of transactions and {best['queue_recall']:.1%} of the fraud")

figure, (left, middle, right) = plt.subplots(1, 3, figsize=(18, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, best["flagged"]), display_labels=["Genuine", "Fraud"]).plot(
    ax=left, cmap="Blues", values_format=",d", colorbar=False)
left.set_title(f"Confusion Matrix (HIGH priority) - {best_name}")
for name in MODEL_NAMES:
    p, r, _ = precision_recall_curve(y_test, results[name]["scores"])
    middle.plot(r, p, label=f"{name} (AP = {results[name]['ap']:.3f})")
    fpr, tpr, _ = roc_curve(y_test, results[name]["scores"])
    right.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
middle.axhline(baseline_ap, color="k", linestyle=":", label="Random guessing")
middle.set_xlabel("Recall (share of fraud caught)")
middle.set_ylabel("Precision (share of flags that are fraud)")
middle.set_title("Precision-Recall Curves")
middle.legend(fontsize=8)
right.plot([0, 1], [0, 1], "k:", label="Random guessing")
right.set_xlabel("False positive rate")
right.set_ylabel("True positive rate")
right.set_title("ROC Curves")
right.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(script_folder, "fraudguard_test_results.png"))
plt.show()

with open(os.path.join(script_folder, "fraudguard_metrics.json"), "w") as stream:
    json.dump({"models": {n: {k: v for k, v in r.items() if k not in ("scores", "flagged")} for n, r in results.items()},
               "selected_model": best_name, "high_lines": high_lines, "medium_lines": medium_lines,
               "random_guessing_ap": baseline_ap, "split": SPLIT_NAME,
               "hours": {"train_last": train_last_hour, "validation_last": val_last_hour},
               "dataset_sha256": paysim_hash,
               "note": "Historical, simulated benchmark. Scores set review priority; they do not prove fraud."},
              stream, indent=2)
print("Results saved to fraudguard_metrics.json. Scores set review priority only - they do not prove fraud.")
print("=" * 100)


# ==================================================================================================
# PART 3 - REVIEWING ONE TRANSACTION AT A TIME: PRIORITY, REASONS AND SPEECH SYNTHESIS
# ==================================================================================================

# ---- speech synthesis: the computer reads every result out loud ----
try:
    import pyttsx3                         # offline text-to-speech (pip install pyttsx3)
    voice = pyttsx3.init()                 # start the computer's built-in voice
    voice.setProperty("rate", 165)         # speaking speed in words per minute
except Exception:                          # pyttsx3 not installed or no voice on this computer
    voice = None
    print("Voice not available (run: pip install pyttsx3) - results will only be shown on screen")


def speak(text):
    """Say the text out loud if a voice is available. The program never stops because of a voice problem."""
    if voice is not None:
        try:
            voice.say(text)
            voice.runAndWait()
        except Exception:
            pass


def review_reasons(row, model_scores):
    """Descriptive training-data context, not explanations of the selected model score."""
    reasons = []
    kind, amount, hour = str(transactions.at[row, "type"]), float(transactions.at[row, "amount"]), \
        int(transactions.at[row, "hour_of_day"])
    if fraud_type_share[kind] > 0:
        reasons.append(f"Type {kind}: {fraud_type_share[kind]:.0%} of the fraud in the training hours had this type")
    else:
        reasons.append(f"Type {kind}: no fraud of this type was seen in the training hours")
    amount_rank = np.searchsorted(sorted_train_amounts, amount) / len(sorted_train_amounts)
    if amount_rank >= 0.90:
        reasons.append(f"Amount {amount:,.2f} is larger than {amount_rank:.0%} of training transactions")
    if average_fraud_rate > 0 and hour_fraud_rate[hour] >= 2 * average_fraud_rate:
        reasons.append(f"Hour {hour} of the cycle: fraud rate {hour_fraud_rate[hour] / average_fraud_rate:.0f} "
                       f"times the average")
    if model_scores["Autoencoder"] >= high_lines["Autoencoder"]:              # only clearly unusual transactions
        reasons.append("The autoencoder finds this transaction very unusual compared with genuine transactions")
    return reasons


checked_transactions = []                         # remembers every transaction reviewed (used by the assistant)
fraud_positions = np.flatnonzero(y_test == 1)     # positions of the fraud cases in the test hours (for the demo)
PRIORITY_COLOURS = {"HIGH": "darkred", "MEDIUM": "darkorange", "LOW": "darkgreen"}

print(f"Review queue with the {best_name}: score {HIGH_LINE:.1%} or more = HIGH | {MEDIUM_LINE:.1%} to "
      f"{HIGH_LINE:.1%} = MEDIUM | below {MEDIUM_LINE:.1%} = LOW priority")

while True:   # keep asking until the user chooses to move on
    choice = input("\n[1] Random test transaction   [2] Random FRAUD transaction   [3] Done reviewing  ->  ").strip()
    if choice == "3":
        print(f"Reviewing finished - {len(checked_transactions)} transaction(s) reviewed")
        break
    if choice == "1":
        spot = random.randrange(len(test_rows))           # any test transaction (almost always genuine, like real life)
    elif choice == "2":
        spot = int(random.choice(fraud_positions))        # a known fraud case, to show what happens with fraud
    else:
        print("Please type 1, 2 or 3")
        continue

    row = test_rows[spot]
    model_scores = {name: float(fraud_score(name, X_test[spot:spot + 1])[0]) for name in MODEL_NAMES}
    score = model_scores[best_name]
    priority = "HIGH" if score >= HIGH_LINE else "MEDIUM" if score >= MEDIUM_LINE else "LOW"
    actual = "Fraud" if y_test[spot] == 1 else "Genuine"

    print(f"Transaction #{row}  |  Hour {int(transactions.at[row, 'step'])}  |  {transactions.at[row, 'type']}  |  "
          f"Amount {float(transactions.at[row, 'amount']):,.2f}  |  Actual label: {actual}")
    print(f"Review priority: {priority}  |  Score {score:.1%} ({best_name})")
    print("Review context (descriptive facts, not model feature attribution):")
    for reason in review_reasons(row, model_scores):
        print("   -", reason)
    print("   - Limitation: the score sets review priority; only an analyst can confirm fraud")
    for name in MODEL_NAMES:
        print(f"   {name:<20}: {model_scores[name]:.1%}")

    checked_transactions.append({"row": int(row), "priority": priority, "score": score, "actual": actual,
                                 "amount": float(transactions.at[row, "amount"])})
    speak(f"{priority} priority for review. Score {score * 100:.0f} percent.")

    plt.figure(figsize=(7, 3))
    plt.barh(MODEL_NAMES, [model_scores[name] for name in MODEL_NAMES], color="steelblue")
    plt.axvline(MEDIUM_LINE, color="orange", linestyle="--", label=f"MEDIUM line ({best_name})")
    plt.axvline(HIGH_LINE, color="darkred", linestyle="--", label=f"HIGH line ({best_name})")
    plt.xlim(0, 1)
    plt.xlabel("Fraud score")
    plt.title(f"{priority} priority ({score:.1%})  |  Actual: {actual}", color=PRIORITY_COLOURS[priority])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()   # close the window to review the next transaction


# ==================================================================================================
# PART 4 - TIME SERIES ANALYSIS AND FORECAST OF HOURLY TRANSACTIONS AND FRAUD
# ==================================================================================================

print("=" * 100)
print("Time series analysis of transactions and fraud per hour...")

# ---- the time series: one row per hour (step) with the number of transactions and fraud cases ----
all_hours = range(profile["first_hour"], profile["last_hour"] + 1)
hourly = transactions.groupby("step")["isFraud"].agg(["size", "sum"]).reindex(all_hours, fill_value=0)
hourly.columns = ["transactions", "fraud_count"]
hourly["fraud_rate"] = hourly["fraud_count"] / hourly["transactions"].clip(lower=1)
hourly.to_csv(os.path.join(script_folder, "fraudguard_hourly.csv"))      # the hourly aggregate data set

by_hour_of_day = transactions.groupby("hour_of_day")["isFraud"].agg(["size", "sum"]).reindex(range(24), fill_value=0)
days = len(hourly) / 24
print(f"{len(hourly)} hours ({days:.0f} days)  |  busiest hour of the cycle: {int(by_hour_of_day['size'].idxmax())}  |  "
      f"quietest: {int(by_hour_of_day['size'].idxmin())}")
rate_by_hour = by_hour_of_day["sum"] / by_hour_of_day["size"].clip(lower=1)
print("Highest fraud rate by hour of the cycle: " + ", ".join(
    f"hour {h} ({rate_by_hour[h]:.2%})" for h in rate_by_hour.sort_values(ascending=False).index[:3]))
print(f"Average transactions per hour: {hourly['transactions'].mean():,.0f}  |  average fraud cases per hour: "
      f"{hourly['fraud_count'].mean():.1f}")


# ---- four forecasting methods for the next 24 hours ----
def forecast_naive(history, steps=24):
    """Every next hour = the last hour."""
    return np.repeat(history[-1], steps)


def forecast_daily_average(history, steps=24):
    """Every next hour = the average of the last 24 hours."""
    return np.repeat(history[-24:].mean(), steps)


def forecast_seasonal_naive(history, steps=24):
    """Every next hour = the same hour one day earlier."""
    return np.resize(history[-24:], steps)


def forecast_seasonal_average(history, steps=24):
    """Every next hour = the average of the same hour over the last 7 days."""
    week = history[-24 * 7:].reshape(-1, 24)
    return np.resize(week.mean(axis=0), steps)


methods = {"Naive (last hour)": forecast_naive, "24-hour average": forecast_daily_average,
           "Seasonal naive (same hour yesterday)": forecast_seasonal_naive,
           "Seasonal average (same hour, last 7 days)": forecast_seasonal_average}

# ---- Forecast method selection: seven daily origins BEFORE the final seven-day holdout ----
forecasts, best_methods, forecast_results = {}, {}, {}
SERIES_LABELS = {"transactions": "transactions", "fraud_count": "fraud cases"}
print("Method selection: 7 daily 24-hour forecasts in the week BEFORE the final week; the chosen method is then")
print("frozen and tested on the final 7 days (holdout), next to the naive forecast.")
for series_name in ("transactions", "fraud_count"):
    series = hourly[series_name].to_numpy(dtype=float)
    if len(series) < 24 * 21:
        raise ValueError("Forecast comparison requires at least 21 days of hourly data.")
    holdout_start = len(series) - 7 * 24
    errors = {}
    for name, method in methods.items():
        errors[name] = float(np.mean([
            np.abs(method(series[:cut]) - series[cut:cut + 24]).mean()
            for cut in range(holdout_start - 7 * 24, holdout_start, 24)]))
    chosen = min(errors, key=errors.get)   # freeze method before inspecting final holdout
    best_methods[series_name] = chosen
    # Daily walk-forward holdout: earlier observed holdout hours become available at each origin.
    test_errors = [float(np.abs(methods[chosen](series[:cut]) - series[cut:cut + 24]).mean())
                   for cut in range(holdout_start, len(series), 24)]
    naive_errors = [float(np.abs(forecast_naive(series[:cut]) - series[cut:cut + 24]).mean())
                    for cut in range(holdout_start, len(series), 24)]
    forecast_results[series_name] = {"selected_method": chosen, "validation_mae": errors,
        "holdout_mae": float(np.mean(test_errors)), "holdout_naive_mae": float(np.mean(naive_errors)),
        "holdout_start_hour": int(hourly.index[holdout_start]), "holdout_hours": 168,
        "protocol": "24-hour forecasts at daily origins; method frozen before holdout"}
    forecasts[series_name] = np.maximum(methods[chosen](series), 0)
    print(f"\nForecasting {SERIES_LABELS[series_name]} per hour")
    print(f"{'Method':<44}{'Selection MAE':>15}")
    for name, value in errors.items():
        print(f"{name:<44}{value:>15,.2f}{'   <- chosen' if name == chosen else ''}")
    print(f"Final-week holdout MAE: {forecast_results[series_name]['holdout_mae']:,.2f} (chosen method) vs "
          f"{forecast_results[series_name]['holdout_naive_mae']:,.2f} (naive forecast)")
    print(f"Next 24 hours: about {forecasts[series_name].sum():,.0f} {SERIES_LABELS[series_name]}")
with open(os.path.join(script_folder, "fraudguard_forecast_metrics.json"), "w") as stream:
    json.dump(forecast_results, stream, indent=2)

# ---- South African context: SABRIC digital banking crime (reported incidents and gross losses) ----
SABRIC_YEARS = ["2022", "2023", "2024", "2025"]
SABRIC_INCIDENTS = np.array([36178, 52584, 97975, 110074])          # reported digital banking crime incidents
SABRIC_LOSSES_MILLION = np.array([734.7, 1082.4, 1888.0, 2400.0])   # gross losses in rand million (2025: about R2.4bn)
# sources: SABRIC Annual Crime Statistics 2023 (2022, 2023), 2024 (2024) and the 2025 report release (2025)
growth = (SABRIC_INCIDENTS[1:] / SABRIC_INCIDENTS[:-1] - 1) * 100
print("\nSouth Africa - digital banking crime (SABRIC):")
for year, cases, losses in zip(SABRIC_YEARS, SABRIC_INCIDENTS, SABRIC_LOSSES_MILLION):
    print(f"   {year}: {cases:>8,} incidents   about R{losses:,.0f} million gross losses")
print("   Incidents grew " + ", ".join(f"{g:.0f}%" for g in growth) + " year on year")

# ---- graphs ----
figure, ((top_left, top_right), (bottom_left, bottom_right)) = plt.subplots(2, 2, figsize=(16, 9))
next_hours = range(profile["last_hour"] + 1, profile["last_hour"] + 25)
top_left.plot(hourly.index, hourly["transactions"], color="steelblue", linewidth=0.8, label="Transactions per hour")
top_left.plot(next_hours, forecasts["transactions"], color="darkred", linestyle="--",
              label=f"Forecast ({best_methods['transactions']})")
top_left.set_title("Transactions per hour and next-24-hour forecast")
top_left.set_xlabel("Hour (step)")
top_left.legend(fontsize=8)
top_left.grid(alpha=0.3)

bottom_left.plot(hourly.index, hourly["fraud_count"], color="darkred", linewidth=0.8, label="Fraud cases per hour")
bottom_left.plot(next_hours, forecasts["fraud_count"], color="black", linestyle="--",
                 label=f"Forecast ({best_methods['fraud_count']})")
bottom_left.set_title("Fraud cases per hour and next-24-hour forecast")
bottom_left.set_xlabel("Hour (step)")
bottom_left.legend(fontsize=8)
bottom_left.grid(alpha=0.3)

top_right.bar(range(24), by_hour_of_day["size"] / days, color="steelblue", label="Average transactions")
top_right.set_xlabel("Hour of the 24-hour cycle")
top_right.set_ylabel("Average transactions")
rate_axis = top_right.twinx()                                        # second y-axis for the fraud rate
rate_axis.plot(range(24), rate_by_hour * 100, color="darkred", marker="o", label="Fraud rate %")
rate_axis.set_ylabel("Fraud rate (%)")
top_right.set_title("Daily pattern: volume and fraud rate by hour")

bottom_right.bar(SABRIC_YEARS, SABRIC_INCIDENTS, color="steelblue")
for position, (cases, losses) in enumerate(zip(SABRIC_INCIDENTS, SABRIC_LOSSES_MILLION)):
    bottom_right.text(position, cases, f"R{losses / 1000:.1f}bn", ha="center", va="bottom", fontsize=9)
bottom_right.set_ylabel("Reported incidents")
bottom_right.set_title("South Africa: digital banking crime (SABRIC)")

figure.suptitle("FraudGuard AI - Time Series Analysis and Forecast")
plt.tight_layout()
plt.savefig(os.path.join(script_folder, "fraudguard_time_series.png"))
plt.show()
print("=" * 100)


# ==================================================================================================
# PART 5 - TEXT ANALYSIS (SUSPICIOUS SMS CLASSIFIER) AND THE ASK FRAUDGUARD FAQ ASSISTANT
# ==================================================================================================

import re                                                              # regular expressions for cleaning text
import csv                                                             # reading the tab-separated SMS file
from sklearn.model_selection import train_test_split                   # stratified split of the messages
from sklearn.feature_extraction.text import TfidfVectorizer            # turns text into numbers (TF-IDF)
from sklearn.naive_bayes import MultinomialNB                          # text model 1: Naive Bayes
from sklearn.metrics.pairwise import linear_kernel                     # cosine similarity for the FAQ search


def normalise(text):
    """Lower case, numbers replaced by # and extra spaces removed - so message templates look the same."""
    text = re.sub(r"\d+", "#", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


# ---- 6A: the SMS spam classifier (NLP) ----
sms_file = None
for current, sub_folders, files in os.walk(downloads_folder):         # UCI file name, or the Kaggle copy spam.csv
    for name in files:
        if name in ("SMSSpamCollection", "spam.csv"):
            sms_file = os.path.join(current, name)
            break
    if sms_file:
        break

sms_model, sms_vectorizer, sms_line, sms_results = None, None, None, None
if sms_file is None:
    print("SMS Spam Collection not found in Downloads - the message checker is skipped (the assistant still works)")
else:
    print("SMS Spam Collection found:", sms_file)
    if os.path.basename(sms_file) == "spam.csv":                       # Kaggle copy: columns v1 (label), v2 (text)
        messages = pd.read_csv(sms_file, encoding="latin-1", usecols=[0, 1])
        messages.columns = ["label", "text"]
    else:                                                              # UCI copy: label <tab> text, no header
        messages = pd.read_csv(sms_file, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE,
                               encoding="utf-8")
    messages = messages.dropna()
    messages = messages[messages["label"].isin(["ham", "spam"])]
    total_messages = len(messages)
    messages["template"] = messages["text"].map(normalise)
    messages = messages.drop_duplicates("template")                    # one copy of each template only
    print(f"Messages: {total_messages:,}  |  after removing duplicates and repeated templates: {len(messages):,} "
          f"({(messages['label'] == 'spam').sum():,} spam)")

    text_labels = (messages["label"] == "spam").astype(int).to_numpy()
    text_train, text_rest, label_train, label_rest = train_test_split(
        messages["text"].to_numpy(), text_labels, test_size=0.30, stratify=text_labels, random_state=SEED)
    text_val, text_test, label_val, label_test = train_test_split(
        text_rest, label_rest, test_size=0.50, stratify=label_rest, random_state=SEED)

    sms_vectorizer = TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    T_train = sms_vectorizer.fit_transform(text_train)                 # TF-IDF learned from training messages only
    T_val, T_test = sms_vectorizer.transform(text_val), sms_vectorizer.transform(text_test)

    text_models = {"Naive Bayes": MultinomialNB(alpha=0.5).fit(T_train, label_train),
                   "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000,
                                                             random_state=SEED).fit(T_train, label_train)}
    text_val_ap = {n: average_precision_score(label_val, m.predict_proba(T_val)[:, 1]) for n, m in text_models.items()}
    sms_name = max(text_val_ap, key=text_val_ap.get)                   # chosen on validation data
    sms_model = text_models[sms_name]
    p, r, lines = precision_recall_curve(label_val, sms_model.predict_proba(T_val)[:, 1])
    sms_line = float(lines[int(np.argmax((2 * p * r / (p + r + 1e-12))[:-1]))])   # best F1 on validation
    test_scores = sms_model.predict_proba(T_test)[:, 1]
    test_flags = test_scores >= sms_line
    sms_results = {"model": sms_name, "line": sms_line, "ap": float(average_precision_score(label_test, test_scores)),
                   "precision": float(precision_score(label_test, test_flags, zero_division=0)),
                   "recall": float(recall_score(label_test, test_flags, zero_division=0)),
                   "f1": float(f1_score(label_test, test_flags, zero_division=0)),
                   "test_messages": int(len(label_test)), "messages_after_dedup": int(len(messages))}
    print(f"Text model: {sms_name}  |  test precision {sms_results['precision']:.1%}, recall {sms_results['recall']:.1%}, "
          f"F1 {sms_results['f1']:.3f}, AP {sms_results['ap']:.3f}")
    with open(os.path.join(script_folder, "fraudguard_text_metrics.json"), "w") as stream:
        json.dump({**sms_results, "sms_sha256": file_fingerprint(sms_file),
                   "note": "Spam-message classifier. A spam label is not a confirmed financial scam."}, stream, indent=2)

    if hasattr(sms_model, "coef_"):                                    # how strongly each word points to spam
        word_weights = sms_model.coef_[0]
    else:
        word_weights = sms_model.feature_log_prob_[1] - sms_model.feature_log_prob_[0]
    vocabulary = sms_vectorizer.get_feature_names_out()


def check_message(text):
    """Spam score and the words that pushed the score up."""
    vector = sms_vectorizer.transform([text])
    score = float(sms_model.predict_proba(vector)[0, 1])
    contributions = vector.toarray()[0] * word_weights
    top = [vocabulary[i] for i in np.argsort(contributions)[::-1][:5] if contributions[i] > 0]
    return score, top


# ---- 6B: the FAQ collection (approved answers from reviewed sources) ----
faq_file = os.path.join(script_folder, "fraudguard_faq.csv")
faq = pd.read_csv(faq_file) if os.path.exists(faq_file) else None
if faq is None or not {"faq_id", "question", "approved_answer", "source_url", "reviewed_on"} <= set(faq.columns):
    print("fraudguard_faq.csv is missing or incomplete - put it next to this python file. FAQ answers are switched off.")
    faq = None
else:
    faq_vectorizer = TfidfVectorizer(preprocessor=normalise, stop_words="english")
    faq_matrix = faq_vectorizer.fit_transform(faq["question"])         # TF-IDF of the approved questions
    print(f"FAQ collection loaded: {len(faq)} approved answers")
FAQ_MATCH_LINE = 0.30                                                  # below this similarity -> fallback answer


def answer_results():
    return (f"The best transaction model is the {best_name}. On the final approximately 20% of transactions (whole hours), which it never saw, its average "
            f"precision is {best['ap']:.3f} against {baseline_ap:.4f} for random guessing. The review queue holds "
            f"{best['queue_share']:.2%} of transactions and {best['queue_recall']:.0%} of the fraud, and the top 1% of "
            f"scores contain {best['top_1_percent_recall']:.0%} of the fraud.")


def answer_models():
    return "We compared four models. " + ". ".join(
        f"{n}: average precision {results[n]['ap']:.3f}" for n in MODEL_NAMES) + f". The {best_name} was chosen on validation data."


def answer_forecast():
    return (f"For the next 24 hours the {best_methods['transactions'].lower()} method expects about "
            f"{forecasts['transactions'].sum():,.0f} transactions and the {best_methods['fraud_count'].lower()} method "
            f"about {forecasts['fraud_count'].sum():,.0f} fraud cases. This is based on simulated data.")


def answer_session():
    if not checked_transactions:
        return "No transactions were reviewed in this session yet."
    counts = {p: sum(t["priority"] == p for t in checked_transactions) for p in ("HIGH", "MEDIUM", "LOW")}
    last = checked_transactions[-1]
    return (f"{len(checked_transactions)} transaction(s) reviewed: {counts['HIGH']} high, {counts['MEDIUM']} medium and "
            f"{counts['LOW']} low priority. The last one was {last['priority']} priority with a score of "
            f"{last['score']:.0%}; its label was {last['actual']}.")


def answer_south_africa():
    return (f"According to SABRIC, reported digital banking crime incidents rose from {SABRIC_INCIDENTS[0]:,} in 2022 to "
            f"{SABRIC_INCIDENTS[-1]:,} in 2025, and losses grew to about R2.4 billion in 2025. FraudGuard was trained on "
            f"simulated data, so a South African bank would validate it on its own data first.")


def answer_text_model():
    if sms_results is None:
        return "The message checker is not available because the SMS Spam Collection was not found."
    return (f"The message checker uses {sms_results['model']} with TF-IDF. On test messages it found "
            f"{sms_results['recall']:.0%} of the spam, and {sms_results['precision']:.0%} of its flags were spam. "
            f"A spam label is a warning, not proof of a scam.")


LIVE_ANSWERS = [   # questions about this program's own results are answered with live numbers
    (("accuracy", "accurate", "how good", "results", "score", "precision", "recall"), answer_results),
    (("compare", "models", "which model", "autoencoder", "random forest", "neural"), answer_models),
    (("forecast", "next 24", "tomorrow", "busy", "predict"), answer_forecast),
    (("session", "summary", "reviewed"), answer_session),
    (("statistic", "sabric", "how big", "south africa"), answer_south_africa),
    (("message checker", "spam model", "text model", "sms model"), answer_text_model)]


def assistant_reply(question):
    """Live answer for project questions, otherwise the closest approved FAQ answer, otherwise a safe fallback."""
    text = normalise(question)
    if re.search(r"\b(account balance|reset my|transfer money for me)\b", text):          # banking actions
        return "I cannot access accounts or perform banking actions. Ask your bank through its official channels.", None
    if re.search(r"\b(weather|recipe|football|capital of)\b", text):                     # clearly off-topic
        return ("I only answer fraud-awareness questions and questions about FraudGuard's results. Try asking about "
                "OTPs, SIM swaps, phishing, safe accounts or reporting fraud."), None
    if "otp" in text and re.search(r"\b(did not|never|didnt|unexpected|unrequested)\b", text) and faq is not None:
        hit = faq.loc[faq["faq_id"] == "F04"]
        if len(hit):
            row = hit.iloc[0]
            return row["approved_answer"], row["source_url"]
    for key_words, answer in LIVE_ANSWERS:
        if any(re.search(r"\b" + re.escape(word), text) for word in key_words):
            return answer(), "FraudGuard results (this run)"
    if faq is not None:
        similarity = linear_kernel(faq_vectorizer.transform([question]), faq_matrix).ravel()   # cosine similarity
        best_faq = int(np.argmax(similarity))
        if similarity[best_faq] >= FAQ_MATCH_LINE:
            return faq.at[best_faq, "approved_answer"], faq.at[best_faq, "source_url"]
    return ("I do not have an approved answer for that. Please contact your bank on the number on the back of your card "
            "or on its official website. Try asking about OTPs, SIM swaps, phishing, safe accounts or reporting fraud."), None


print("\nText analysis and Ask FraudGuard (FraudGuard explains and warns - it does not confirm fraud)")
speak("Hello, I am FraudGuard. Check a message or ask me a question.")
while True:
    choice = input("\n[1] Check a message   [2] Ask FraudGuard   [3] Exit  ->  ").strip()
    if choice == "3":
        print("Goodbye. Thank you for using FraudGuard AI")
        speak("Goodbye. Thank you for using FraudGuard AI")
        break
    if choice == "1":
        if sms_model is None:
            print("The message checker is not available (SMS Spam Collection not found)")
            continue
        message = input("Paste the SMS text:   ").strip()
        score, words = check_message(message)
        verdict = "SUSPICIOUS" if score >= sms_line else "No strong spam signs"
        print(f"{verdict}  |  spam score {score:.1%}  (line {sms_line:.1%}, {sms_results['model']})")
        if words and score >= sms_line:
            print("Suspicious wording:", ", ".join(words))
        print("Reminder: never share an OTP, PIN or password, and contact your bank only on its official number.")
        speak(f"{verdict}. Spam score {score * 100:.0f} percent.")
    elif choice == "2":
        question = input("You:   ").strip()
        if question.lower() in ("hi", "hello", "help", ""):
            reply, source = ("Ask me about OTPs, SIM swaps, phishing, vishing, safe-account scams, reporting fraud, "
                             "the model results, the forecast or South African fraud statistics."), None
        else:
            reply, source = assistant_reply(question)
        print("FraudGuard:", reply)
        if source:
            print("Source:", source)
        speak(reply)
    else:
        print("Please type 1, 2 or 3")

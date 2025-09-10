# TASK - 3: DECISION TREE CLASSIFIER
# DATASET: BANK MARKETING DATASET (BANK-FULL.CSV)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

file_path = (r"C:\Users\MYTHRI  MR\OneDrive\Desktop\project\bank-full.csv") 
data = pd.read_csv(file_path, sep=";")
X = data.drop("y", axis=1)
y = data["y"]

# ENCODE TARGET VARIABLE
y = LabelEncoder().fit_transform(y)  # YES=1, NO=0

# IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

# PREPROCESSING: ONE-HOT ENCODE CATEGORICAL VARIABLES
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# BUILD PIPELINE WITH DECISION TREE
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# TRAIN MODEL
clf.fit(X_train, y_train)

# PREDICTIONS
y_pred = clf.predict(X_test)

# EVALUATION
print("\nMODEL PERFORMANCE:")
print("ACCURACY:", accuracy_score(y_test, y_pred))
print("\nCONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred, target_names=["No", "Yes"]))


# EXTRACT THE TRAINED DECISION TREE 
dt_model = clf.named_steps["classifier"]

# TREE PLOT
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    filled=True,
    feature_names=clf.named_steps["preprocessor"].get_feature_names_out(),
    class_names=["No", "Yes"],
    max_depth=3
)
plt.show()

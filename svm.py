import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Generate a synthetic binary dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_informative=3, 
    random_state=42
)

# 2. Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Scale features (SVM is sensitive to data scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and train the SVM Classifier
# 'linear', 'rbf' (default), 'poly', or 'sigmoid' can be selected as the kernel
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 5. Make predictions on the test data
y_pred = svm_model.predict(X_test_scaled)

# 6. Evaluate the model performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

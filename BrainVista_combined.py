# ===== File: BrainVista.py =====
#!/usr/bin/env python
# coding: utf-8

# # **Machine Learning Models**
# 

# **Support Vector Machine using RBF Kernal**
# 

# In[ ]:


from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Define dataset paths
normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

# Check if folders exist
print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def load_images_from_folder(folder, label, img_size=(227, 227), max_images=950):
    images = []
    labels = []
    filenames = os.listdir(folder)

    random.shuffle(filenames)
    filenames = filenames[:max_images]

    for filename in tqdm(filenames, desc=f"Loading {os.path.basename(folder)}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


normal_images, normal_labels = load_images_from_folder(normal_path, label=0, max_images=950)
stroke_images, stroke_labels = load_images_from_folder(stroke_path, label=1, max_images=950)

X = np.concatenate((normal_images, stroke_images))
y = np.concatenate((normal_labels, stroke_labels))

X = X.reshape(X.shape[0], -1)

print(f"Final Dataset Size: {X.shape[0]} images")


# In[ ]:


# Split dataset into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print dataset sizes
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")


# In[ ]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

# Train model
svm_model.fit(X_train, y_train)

print("SVM Model Trained Successfully!")


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report

y_val_pred = svm_model.predict(X_val)
y_test_pred = svm_model.predict(X_test)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.2f}")

print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred))



# **Logistic Regression**
# 
# 

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"
import os
if not os.path.exists(normal_path) or not os.path.exists(stroke_path):
    raise FileNotFoundError("Dataset paths not found. Check Google Drive directory.")


# In[ ]:


import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def load_images_from_folder(folder, label, img_size=(227, 227), max_images=950):
    images = []
    labels = []
    filenames = os.listdir(folder)
    random.shuffle(filenames)
    filenames = filenames[:max_images]

    for filename in tqdm(filenames, desc=f"Loading {os.path.basename(folder)}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Converting  into grayscale

        if img is not None:
            img = cv2.resize(img, img_size)  # Resizing the image to 227x227
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

normal_images, normal_labels = load_images_from_folder(normal_path, label=0, max_images=950)
stroke_images, stroke_labels = load_images_from_folder(stroke_path, label=1, max_images=950)
X = np.concatenate((normal_images, stroke_images))
y = np.concatenate((normal_labels, stroke_labels))
X = X.reshape(X.shape[0], -1)

print(f"Final Dataset Size: {X.shape[0]} images")


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training Set: {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

scaler = MinMaxScaler() # Because we are using gradient descend which needs to be scaled

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(C=0.0006, solver='saga', max_iter=50, penalty='l2', random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("Logistic Regression Model Trained Successfully!")


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report

y_val_pred = lr_model.predict(X_val)
y_test_pred = lr_model.predict(X_test)

val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")
print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))

print(f"Testing Accuracy: {test_accuracy:.2f}")
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred))


# **Decision Tree**
# 

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)
normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

def load_images_from_folder(folder, label, img_size=(227, 227), max_images=950):
    images = []
    labels = []
    filenames = os.listdir(folder)

    random.shuffle(filenames)
    filenames = filenames[:max_images]

    for filename in tqdm(filenames, desc=f"Loading {os.path.basename(folder)}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

normal_images, normal_labels = load_images_from_folder(normal_path, label=0, max_images=950)
stroke_images, stroke_labels = load_images_from_folder(stroke_path, label=1, max_images=950)
X = np.concatenate((normal_images, stroke_images))
y = np.concatenate((normal_labels, stroke_labels))

X = X.reshape(X.shape[0], -1)

print(f"Final Dataset Size: {X.shape[0]} images")


# In[ ]:


# Spliting dataset into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training Set: {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=11, random_state=42)

# Training the model
dt_model.fit(X_train, y_train)
print("Decision Tree Model Trained Successfully!")


# In[ ]:


# Predictions on validation and test sets
y_val_pred_dt = dt_model.predict(X_val)
y_test_pred_dt = dt_model.predict(X_test)

val_accuracy_dt = accuracy_score(y_val, y_val_pred_dt)
print(f"Validation Accuracy (Decision Tree): {val_accuracy_dt:.2f}")

print("Validation Set Evaluation (Decision Tree):")
print(classification_report(y_val, y_val_pred_dt))

test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt)
print(f"Testing Accuracy (Decision Tree): {test_accuracy_dt:.2f}")

print("Test Set Evaluation (Decision Tree):")
print(classification_report(y_test, y_test_pred_dt))


# **Random Forest**

# In[ ]:


from google.colab import drive
import os
drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def load_images_from_folder(folder, label, img_size=(227, 227), max_images=950):
    images = []
    labels = []
    filenames = os.listdir(folder)
    random.shuffle(filenames)
    filenames = filenames[:max_images]

    for filename in tqdm(filenames, desc=f"Loading {os.path.basename(folder)}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

        if img is not None:
            img = cv2.resize(img, img_size)  # Resize to 227x227
            images.append(img)
            labels.append(label)  # Assign label

    return np.array(images), np.array(labels)

normal_images, normal_labels = load_images_from_folder(normal_path, label=0, max_images=950)
stroke_images, stroke_labels = load_images_from_folder(stroke_path, label=1, max_images=950)

X = np.concatenate((normal_images, stroke_images))
y = np.concatenate((normal_labels, stroke_labels))

X = X.reshape(X.shape[0], -1)  # Flatten images into 1D arrays

print(f"Final Dataset Size: {X.shape[0]} images")


# In[ ]:


# Split dataset into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print dataset sizes
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")


# In[ ]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=100,         # Reduce number of trees further
    max_depth=12,            # Reduce tree depth
    min_samples_split=10,    # Require more samples to split
    min_samples_leaf=5,      # Increase minimum leaf size
    max_features='sqrt',     # Use even fewer features per split
    random_state=42,
    n_jobs=-1                # Use all CPU cores for faster training
)
rf_model.fit(X_train_scaled, y_train)
print("Random Forest Model Trained Successfully!")


# In[ ]:


# Predictions on validation and test sets
y_val_pred = rf_model.predict(X_val_scaled)
y_test_pred = rf_model.predict(X_test_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.3f}")

print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.3f}")

print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred))


# **Naive Bayes**

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def load_images_from_folder(folder, label, img_size=(227, 227), max_images=950):
    images = []
    labels = []

    filenames = os.listdir(folder)

    random.shuffle(filenames)
    filenames = filenames[:max_images]

    for filename in tqdm(filenames, desc=f"Loading {os.path.basename(folder)}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, img_size)  # Resize to 227x227
            images.append(img)
            labels.append(label)  # Assign label

    return np.array(images), np.array(labels)

normal_images, normal_labels = load_images_from_folder(normal_path, label=0, max_images=950)
stroke_images, stroke_labels = load_images_from_folder(stroke_path, label=1, max_images=950)

X = np.concatenate((normal_images, stroke_images))
y = np.concatenate((normal_labels, stroke_labels))

X = X.reshape(X.shape[0], -1)  # Flatten images into 1D arrays

print(f"Final Dataset Size: {X.shape[0]} images")


# In[ ]:


# Split dataset into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print dataset sizes
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report

# Apply Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Remove constant features
variance_filter = VarianceThreshold(threshold=0.0001)  # Removes low-variance features
X_train_var = variance_filter.fit_transform(X_train_scaled)
X_val_var = variance_filter.transform(X_val_scaled)
X_test_var = variance_filter.transform(X_test_scaled)

selector = SelectKBest(f_classif, k=10000)
X_train_selected = selector.fit_transform(X_train_var, y_train)
X_val_selected = selector.transform(X_val_var)
X_test_selected = selector.transform(X_test_var)

pca = PCA(n_components=200)
X_train_pca = pca.fit_transform(X_train_selected)
X_val_pca = pca.transform(X_val_selected)
X_test_pca = pca.transform(X_test_selected)

nb_model = GaussianNB(var_smoothing=1e-5)  # Slightly increased smoothing for stability
nb_model.fit(X_train_pca, y_train)

print("Naïve Bayes Model Trained Successfully!")


# In[ ]:


# Predictions on validation and test sets
y_val_pred = nb_model.predict(X_val_pca)
y_test_pred = nb_model.predict(X_test_pca)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.3f}")

print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.3f}")

print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred))


# # **Deep Learning Models**

# **DNN Model**

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score


# In[ ]:


def load_images(folder_path, label, image_size=(227, 227)):
    images, labels = [], []
    for i, file in enumerate(os.listdir(folder_path)):
        if i >= 950:
            break
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img / 255.0)
                labels.append(label)
    return images, labels

normal_imgs, normal_labels = load_images(normal_path, 0)
stroke_imgs, stroke_labels = load_images(stroke_path, 1)

X = np.array(normal_imgs + stroke_imgs, dtype=np.float32).reshape(-1, 227, 227, 1)
y = np.array(normal_labels + stroke_labels)

print(f"Total Samples: {len(X)} | Normal: {len(normal_imgs)} | Stroke: {len(stroke_imgs)}")


# In[ ]:


# DNN Model Architecture
def create_dnn_model():
    model = Sequential([
        Input(shape=(227, 227, 1)),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


# In[ ]:


fold_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_idx, val_idx in skf.split(X, y):
    print(f"\nStarting Fold {fold}...\n" + "-"*50)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_dnn_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f'dnn_best_fold_{fold}.keras', save_best_only=True, monitor='val_loss', verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype("int32")

    acc = np.mean(y_pred.flatten() == y_val)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Final Report
    print(f"\nFold {fold} Results:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {roc_auc:.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))

    # ROC Curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold}")
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    fold_results.append({
        'fold': fold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': roc_auc
    })

    fold += 1


# In[ ]:


best_fold_auc = max(fold_results, key=lambda x: x['auc'])

print("\nBest Fold Based on AUC:")
print(f"Fold     : {best_fold_auc['fold']}")
print(f"Accuracy : {best_fold_auc['accuracy']:.4f}")
print(f"Precision: {best_fold_auc['precision']:.4f}")
print(f"Recall   : {best_fold_auc['recall']:.4f}")
print(f"F1 Score : {best_fold_auc['f1']:.4f}")
print(f"AUC      : {best_fold_auc['auc']:.4f}")


# **Artificial Neural Network (ANN)**

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


# In[ ]:


def load_images(folder_path, label, image_size=(227, 227), limit=None):
    images, labels = [], []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            images.append(img / 255.0)
            labels.append(label)
            if limit and len(images) >= limit:
                break
    return np.array(images), np.array(labels)

# Set the number of images to be 950 for each class
num_images = 950

# Load exactly 950 images for each class
X_normal, y_normal = load_images(normal_path, 0, limit=num_images)
X_stroke, y_stroke = load_images(stroke_path, 1, limit=num_images)

# Combine the normal and stroke images
X = np.concatenate([X_normal, X_stroke])
y = np.concatenate([y_normal, y_stroke])

# Reshape for ANN
X = X.reshape(-1, 227, 227, 1)

print(f"Final Dataset Shape: {X.shape} | Labels: {y.shape}")


# In[ ]:


def build_ann(input_shape=(227, 227, 1)):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[
                      'accuracy',
                      tf.keras.metrics.AUC(name='auc'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')
                  ])
    return model


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# List to store fold results
fold_results = []

for train_idx, val_idx in skf.split(X, y):
    print(f"\nFold {fold} Training...\n{'-'*40}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_ann()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Predictions
    y_pred_proba = model.predict(X_val).flatten()
    y_pred = (y_pred_proba > 0.5).astype("int32")

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Store fold results
    fold_results.append({
        'fold': fold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': roc_auc
    })

    print(f"\nFold {fold} Results:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {roc_auc:.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))

    # Plot ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend()
    plt.show()

    fold += 1

# Step 1: Choose the best fold based on AUC (or any other metric)
best_fold_auc = max(fold_results, key=lambda x: x['auc'])

# Step 2: Output the best fold details
print("\nBest Fold Based on AUC:")
print(f"Fold {best_fold_auc['fold']} — AUC: {best_fold_auc['auc']:.4f}")
print(f"Accuracy: {best_fold_auc['accuracy']:.4f}")
print(f"Precision: {best_fold_auc['precision']:.4f}")
print(f"Recall: {best_fold_auc['recall']:.4f}")
print(f"F1 Score: {best_fold_auc['f1_score']:.4f}")


# **Recurrent Nueral Network**

# In[ ]:


from google.colab import drive
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"
print("Normal Path:", os.path.exists(normal_path))
print("Stroke Path:", os.path.exists(stroke_path))


# In[ ]:


def load_images(folder_path, label, image_size=(227, 227), limit=950):
    images, labels = [], []
    count = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            img = img / 255.0
            images.append(img)
            labels.append(label)
            count += 1
            if count == limit:
                break
    return images, labels
normal_imgs, normal_labels = load_images(normal_path, 0, limit=950)
stroke_imgs, stroke_labels = load_images(stroke_path, 1, limit=950)

X = np.array(normal_imgs + stroke_imgs)
y = np.array(normal_labels + stroke_labels)
X_rnn = X.reshape(-1, 227, 227)

print("Total Samples:", len(X_rnn), "| Normal:", len(normal_imgs), "| Stroke:", len(stroke_imgs))


# In[ ]:


def create_rnn_model():
    model = Sequential([
        Input(shape=(227, 227)),
        SimpleRNN(128, activation='tanh',return_sequences=False),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
fold_results = []

for train_idx, val_idx in skf.split(X_rnn, y):
    print(f"\nFold {fold} — Training...\n" + "-"*50)

    X_train, X_val = X_rnn[train_idx], X_rnn[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_rnn_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f"rnn_best_fold_{fold}.keras", save_best_only=True, monitor='val_loss')
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype("int32")

    # Calculate metrics
    acc = np.mean(y_pred.flatten() == y_val)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Store fold results
    fold_results.append({
        "fold": fold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": roc_auc
    })

    # Print fold results
    print(f"\nFold {fold} Results:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {roc_auc:.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))

    # Plot ROC Curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.2f})', color='darkblue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold}")
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    fold += 1


# In[ ]:


best_fold_auc = max(fold_results, key=lambda x: x['auc'])

print("\nBest Fold Based on AUC:")
print(f"Fold {best_fold_auc['fold']} — AUC: {best_fold_auc['auc']:.4f}")
print(f"Accuracy: {best_fold_auc['accuracy']:.4f}")
print(f"Precision: {best_fold_auc['precision']:.4f}")
print(f"Recall: {best_fold_auc['recall']:.4f}")
print(f"F1 Score: {best_fold_auc['f1_score']:.4f}")


# # CNN(using Feature Extraction)
# 

# Modified AlexNET

# In[ ]:


# 1. Mount Drive and check paths
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


# 2. Load and Preprocess Images
import cv2
import numpy as np
from tqdm import tqdm

def load_balanced_images(normal_folder, stroke_folder, img_size=(227, 227)):
    images, labels = [], []
    max_images = 950  # Balance both classes

    for folder, label in [(normal_folder, 0), (stroke_folder, 1)]:
        count = 0
        files = os.listdir(folder)
        files.sort()  # Optional: For consistent order
        for filename in tqdm(files[:max_images], desc=f'Loading {"Normal" if label==0 else "Stroke"}'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
                count += 1
        print(f"Loaded {count} {'Normal' if label==0 else 'Stroke'} images")

    images = np.array(images).reshape(-1, 227, 227, 1).astype('float32') / 255.0
    labels = np.array(labels)
    return images, labels

X, y = load_balanced_images(normal_path, stroke_path)
print(f"\nFinal Dataset Shape: {X.shape}, Labels: {y.shape}")


# In[ ]:


# 3. Split into Train, Validation, and Test
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training Set:   {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set:    {X_test.shape[0]} samples")


# In[ ]:


# 4. Define the AlexNet-Based Model
import tensorflow as tf
from tensorflow.keras import layers, models, Input

input_shape = (227, 227, 1)
inputs = Input(shape=input_shape)

# Conv1
x = layers.Conv2D(64, (16, 8), padding='same', activation='relu', name='Conv1')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Conv2
x = layers.Conv2D(192, (8, 4), padding='same', activation='relu', name='Conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Conv3-1, 3-2, 3-3
x = layers.Conv2D(384, (4, 4), padding='same', activation='relu', name='Conv3-1')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (4, 4), padding='same', activation='relu', name='Conv3-2')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (4, 4), padding='same', activation='relu', name='Conv3-3')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Special Pooling (4x2)
x = layers.MaxPooling2D(pool_size=(4, 2), strides=(1, 1), padding='valid', name='Pool4-2')(x)

# FC Layers
x = layers.GlobalAveragePooling2D()(x)  # Reduce size before FC
x = layers.Dense(4096, activation='relu', name='Fc4')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(4096, activation='relu', name='Fc5')(x)
outputs = layers.Dense(2, activation='softmax', name='Fc6softmax')(x)


modified_alexnet = models.Model(inputs=inputs, outputs=outputs)
modified_alexnet.summary()


# In[ ]:


# 5. Compile and Train the Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

modified_alexnet.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
]

history = modified_alexnet.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

modified_alexnet.save('stroke_detection_model.h5')
print("Model training complete and saved!")


# In[ ]:


import numpy as np
from sklearn.metrics import classification_report

# Evaluate on test set
test_loss, test_accuracy = modified_alexnet.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# Predict classes
y_pred = modified_alexnet.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# If y_test is already in label format (not one-hot), use directly
y_true_classes = y_test

# Classification report
print("📊 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=["Normal", "Stroke"], zero_division=0))


# In[ ]:


from tensorflow.keras import models
import numpy as np

# Create feature extractor model from the second FC layer (Fc5)
feature_extractor = models.Model(inputs=modified_alexnet.input,
                                 outputs=modified_alexnet.get_layer("Fc5").output)

# Extract features
train_features = feature_extractor.predict(X_train)
val_features = feature_extractor.predict(X_val)
test_features = feature_extractor.predict(X_test)

# Save features as .npy files
np.save("modified_alexnet_features_train.npy", train_features)
np.save("modified_alexnet_features_val.npy", val_features)
np.save("modified_alexnet_features_test.npy", test_features)

print("Modified AlexNet features saved as .npy files successfully.")


# Modified InceptionV3

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


# 2. Load and Preprocess Images for InceptionV3
import cv2
import numpy as np
from tqdm import tqdm

def load_balanced_images(normal_folder, stroke_folder, img_size=(299, 299)):  # InceptionV3 requires 299x299
    images, labels = [], []
    max_images = 950  # Balance both classes

    for folder, label in [(normal_folder, 0), (stroke_folder, 1)]:
        count = 0
        files = os.listdir(folder)
        files.sort()  # Optional: For consistent order
        for filename in tqdm(files[:max_images], desc=f'Loading {"Normal" if label==0 else "Stroke"}'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # InceptionV3 expects RGB
                images.append(img)
                labels.append(label)
                count += 1
        print(f"Loaded {count} {'Normal' if label==0 else 'Stroke'} images")

    images = np.array(images).astype('float32') / 255.0  # Normalize to [0,1]
    labels = np.array(labels)
    return images, labels

X, y = load_balanced_images(normal_path, stroke_path)
print(f"\nFinal Dataset Shape: {X.shape}, Labels: {y.shape}")


# In[ ]:


# 3. Split into Train, Validation, and Test (same as before)
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training Set:   {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set:    {X_test.shape[0]} samples")


# In[ ]:


# 4. Define the InceptionV3-Based Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, models, Input

# Load pre-trained InceptionV3 without top layers
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)  # InceptionV3 expects 299x299 RGB images
)

# Freeze the base model
base_model.trainable = False

# Create new model on top
inputs = Input(shape=(299, 299, 3))
x = base_model(inputs, training=False)

# Add custom layers
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(2, activation='softmax')(x)

inceptionv3_model = models.Model(inputs, outputs)
inceptionv3_model.summary()


# In[ ]:


# 5. Compile and Train the Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

inceptionv3_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint('best_inceptionv3_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
]

history = inceptionv3_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

inceptionv3_model.save('stroke_detection_inceptionv3.h5')
print("InceptionV3 model training complete and saved!")


# In[ ]:


# 6. Evaluate the Model
import numpy as np
from sklearn.metrics import classification_report

# Evaluate on test set
test_loss, test_accuracy = inceptionv3_model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# Predict classes
y_pred = inceptionv3_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=["Normal", "Stroke"], zero_division=0))


# In[ ]:


# 7. Extract and Save Features (from GlobalAveragePooling layer)
from tensorflow.keras import models
import numpy as np

# Create feature extractor model
feature_extractor = models.Model(
    inputs=inceptionv3_model.input,
    outputs=inceptionv3_model.get_layer("global_average_pooling2d_2").output
)

# Extract features
train_features = feature_extractor.predict(X_train)
val_features = feature_extractor.predict(X_val)
test_features = feature_extractor.predict(X_test)

# Save features as .npy files
np.save("inceptionv3_features_train.npy", train_features)
np.save("inceptionv3_features_val.npy", val_features)
np.save("inceptionv3_features_test.npy", test_features)

print("InceptionV3 features saved as .npy files successfully.")


# VGG19

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import cv2
import numpy as np
from tqdm import tqdm

def load_balanced_images(normal_folder, stroke_folder, img_size=(224, 224)):  # VGG19 requires 224x224
    images, labels = [], []
    max_images = 950  # Balance both classes

    for folder, label in [(normal_folder, 0), (stroke_folder, 1)]:
        count = 0
        files = os.listdir(folder)
        files.sort()  # Optional: For consistent order
        for filename in tqdm(files[:max_images], desc=f'Loading {"Normal" if label==0 else "Stroke"}'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # VGG19 expects RGB
                images.append(img)
                labels.append(label)
                count += 1
        print(f"Loaded {count} {'Normal' if label==0 else 'Stroke'} images")

    images = np.array(images).astype('float32') / 255.0  # Normalize to [0,1]
    labels = np.array(labels)
    return images, labels

X, y = load_balanced_images(normal_path, stroke_path)
print(f"\nFinal Dataset Shape: {X.shape}, Labels: {y.shape}")


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training Set:   {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set:    {X_test.shape[0]} samples")


# In[ ]:


# 4. Define the Exact VGG19 Architecture with Global Average Pooling
from tensorflow.keras import layers, models, Input

def build_vgg19_gap(input_shape=(224, 224, 3), num_classes=2):
    inputs = Input(shape=input_shape, name='input_image')

    # Block 1 (2 conv layers @ 64)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2 (2 conv layers @ 128)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3 (4 conv layers @ 256)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4 (4 conv layers @ 512)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5 (4 conv layers @ 512)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Replace Flatten with Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Fully Connected Layers
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='vgg19_gap')

# Create model
vgg19_gap = build_vgg19_gap(input_shape=(224, 224, 3))
vgg19_gap.summary()


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Compile the model
vgg19_gap.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    ModelCheckpoint('best_vgg19_gap.h5',
                   monitor='val_accuracy',
                   save_best_only=True,
                   mode='max'),
    EarlyStopping(monitor='val_loss',
                  patience=7,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.2,
                      patience=3,
                      min_lr=1e-7)
]

# Train the model
history = vgg19_gap.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
vgg19_gap.save('vgg19_gap_final.h5')
print("Training complete and model saved!")


# In[ ]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluate on test set
test_loss, test_acc = vgg19_gap.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# Generate predictions
y_pred = vgg19_gap.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes,
                           target_names=["Normal", "Stroke"],
                           digits=4))



# In[ ]:


from tensorflow.keras import models
import numpy as np

# Create feature extractor from the Global Average Pooling layer
feature_extractor = models.Model(
    inputs=vgg19_gap.input,
    outputs=vgg19_gap.get_layer('global_avg_pool').output
)

# Extract features
print("\nExtracting features...")
train_features = feature_extractor.predict(X_train)
val_features = feature_extractor.predict(X_val)
test_features = feature_extractor.predict(X_test)

# Save features
np.save("vgg19_gap_train_features.npy", train_features)
np.save("vgg19_gap_val_features.npy", val_features)
np.save("vgg19_gap_test_features.npy", test_features)
print("Features saved successfully!")



# ## **NasNETLarge**

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import cv2
import numpy as np
from tqdm import tqdm

def load_balanced_images(normal_folder, stroke_folder, img_size=(331, 331)):  # NASNetLarge requires 331x331
    images, labels = [], []
    max_images = 950  # Balance both classes

    for folder, label in [(normal_folder, 0), (stroke_folder, 1)]:
        count = 0
        files = os.listdir(folder)
        files.sort()  # Optional: For consistent order
        for filename in tqdm(files[:max_images], desc=f'Loading {"Normal" if label==0 else "Stroke"}'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # NASNet expects RGB
                images.append(img)
                labels.append(label)
                count += 1
        print(f"Loaded {count} {'Normal' if label==0 else 'Stroke'} images")

    images = np.array(images).astype('float32') / 255.0  # Normalize to [0,1]
    labels = np.array(labels)
    return images, labels

X, y = load_balanced_images(normal_path, stroke_path)
print(f"\nFinal Dataset Shape: {X.shape}, Labels: {y.shape}")


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training Set:   {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set:    {X_test.shape[0]} samples")


# In[ ]:


from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras import layers, models, Input

# Load pre-trained NASNetLarge without top layers
base_model = NASNetLarge(
    weights='imagenet',
    include_top=False,
    input_shape=(331, 331, 3)  # NASNetLarge expects 331x331 RGB images
)
base_model.trainable = False
inputs = Input(shape=(331, 331, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation='softmax')(x)

nasnet_model = models.Model(inputs, outputs)
nasnet_model.summary()


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

nasnet_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint('best_nasnet_model.h5',
                   monitor='val_accuracy',
                   save_best_only=True,
                   mode='max'),
    EarlyStopping(monitor='val_loss',
                  patience=5,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.2,
                      patience=3,
                      min_lr=1e-7)
]

history = nasnet_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,  # Fewer epochs may be sufficient for transfer learning
    batch_size=8,  # Smaller batch size due to memory constraints
    callbacks=callbacks,
    verbose=1
)

nasnet_model.save('stroke_detection_nasnet.h5')
print("NASNetLarge model training complete and saved!")


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluate on test set
test_loss, test_acc = nasnet_model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# Generate predictions
y_pred = nasnet_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes,
                           target_names=["Normal", "Stroke"],
                           digits=4))



# In[ ]:


# 7. Feature Extraction with Correct Layer Name
from tensorflow.keras import models
import numpy as np

# Create feature extractor using the correct layer name
feature_extractor = models.Model(
    inputs=nasnet_model.input,
    outputs=nasnet_model.get_layer('global_average_pooling2d').output  # Corrected layer name
)

# Extract features
print("\nExtracting features from NASNetLarge...")
train_features = feature_extractor.predict(X_train)
val_features = feature_extractor.predict(X_val)
test_features = feature_extractor.predict(X_test)

# Save features
np.save("nasnet_train_features.npy", train_features)
np.save("nasnet_val_features.npy", val_features)
np.save("nasnet_test_features.npy", test_features)
print("✅ NASNetLarge features extracted and saved successfully!")


# ## **ShuffleNET**

# In[ ]:


from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

normal_path = "/content/drive/MyDrive/Normal"
stroke_path = "/content/drive/MyDrive/Stroke"

print("Normal Path Exists:", os.path.exists(normal_path))
print("Stroke Path Exists:", os.path.exists(stroke_path))


# In[ ]:


import cv2
import numpy as np
from tqdm import tqdm

def load_balanced_images(normal_folder, stroke_folder, img_size=(224, 224)):
    images, labels = [], []
    max_images = 950  # Balance both classes

    for folder, label in [(normal_folder, 0), (stroke_folder, 1)]:
        count = 0
        files = os.listdir(folder)
        files.sort()
        for filename in tqdm(files[:max_images], desc=f'Loading {"Normal" if label==0 else "Stroke"}'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
                count += 1
        print(f"Loaded {count} {'Normal' if label==0 else 'Stroke'} images")

    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels)
    return images, labels

X, y = load_balanced_images(normal_path, stroke_path)
print(f"\nFinal Dataset Shape: {X.shape}, Labels: {y.shape}")


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training Set:   {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")
print(f"Testing Set:    {X_test.shape[0]} samples")


# In[ ]:


get_ipython().system('git clone https://github.com/opconty/keras-shufflenetV2.git')
get_ipython().run_line_magic('cd', 'keras-shufflenetV2')


# In[ ]:


get_ipython().system('rm -f keras-shufflenetV2/shufflenetv2.py')


# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, ReLU, Concatenate, DepthwiseConv2D, Lambda

# Split Channels for Channel Shuffle
def channel_split(x):
    def split_tensor(tensor):
        in_channels = tf.shape(tensor)[-1]
        return tf.split(tensor, num_or_size_splits=2, axis=-1)
    return Lambda(lambda z: split_tensor(z))(x)

# Perform Channel Shuffle
def channel_shuffle(x, groups):
    def shuffle(z):
        shape = tf.shape(z)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        group_channels = channels // groups
        z = tf.reshape(z, [batch_size, height, width, groups, group_channels])
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, channels])
        return z
    return Lambda(shuffle)(x)

# Convolutional branch for ShuffleNetV2
def conv_branch(x, out_channels, stride):
    x = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Shuffle Unit Block
def shuffle_unit(inputs, out_channels, stride):
    if stride == 1:
        split = channel_split(inputs)
        x1 = Lambda(lambda z: z[0])(split)
        x2 = Lambda(lambda z: z[1])(split)
        x2 = conv_branch(x2, out_channels // 2, stride)
        out = Concatenate(axis=-1)([x1, x2])
    else:
        x1 = conv_branch(inputs, out_channels // 2, stride)
        x2 = conv_branch(inputs, out_channels // 2, stride)
        out = Concatenate(axis=-1)([x1, x2])
    return channel_shuffle(out, 2)

# ShuffleNetV2 Feature Extractor Model
def ShuffleNetV2_FeatureExtractor(input_shape=(224, 224, 3), scale_factor=1.0, output_dim=1000):
    input = Input(shape=input_shape)

    # Define channel sizes for various scale factors
    out_channels_dict = {
        0.5: [24, 48, 96, 192, 1024],
        1.0: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2.0: [24, 244, 488, 976, 2048]
    }

    out_channels = out_channels_dict[scale_factor]

    # Initial Convolution Block
    x = Conv2D(out_channels[0], 3, strides=2, padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stages with repeated ShuffleNetV2 blocks
    stages_repeat = [4, 8, 4]  # This can be adjusted based on the scale_factor

    for stage, reps in enumerate(stages_repeat):
        out_ch = out_channels[stage + 1]
        for i in range(reps):
            stride = 2 if i == 0 else 1
            x = shuffle_unit(x, out_ch, stride)

    # Final Convolution and Pooling Block
    x = Conv2D(out_channels[-1], 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)

    # Final Fully Connected Layer
    output = Dense(output_dim, activation='softmax')(x)  # Softmax for multi-class classification

    model = models.Model(inputs=input, outputs=output, name="ShuffleNetV2_FeatureExtractor")
    return model

# Instantiate and summarize the model
model = ShuffleNetV2_FeatureExtractor(input_shape=(224, 224, 3), scale_factor=1.0, output_dim=1000)
model.summary()



# In[ ]:


# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for Model Checkpointing, Early Stopping, and Learning Rate Scheduling
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint('best_shufflenet_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'),
    EarlyStopping(monitor='val_loss',
                  patience=5,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.2,
                      patience=3,
                      min_lr=1e-7)
]

# Train the model (ensure your training data is prepared: X_train, y_train, X_val, y_val)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,  # Adjust the number of epochs if needed
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# Save the trained model
model.save('stroke_detection_shufflenet.h5')
print("ShuffleNet model training complete and saved!")


# In[ ]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes,
                           target_names=["Normal", "Stroke"],  # Replace with your classes if different
                           digits=4))



# In[ ]:


from tensorflow.keras import models
import numpy as np

# Extract features just before the final Dense layer (based on your model's last Conv or Concatenate layer)
# We assume the final layer before classification is 'concatenate_24' (or similar based on your model structure)
feature_extractor = models.Model(
    inputs=model.input,
    outputs=model.get_layer('concatenate_4').output  # Use the correct last Conv or Concatenate layer
)

# Extract features
print("\nExtracting features from ShuffleNetV2...")
train_features = feature_extractor.predict(X_train)
val_features = feature_extractor.predict(X_val)
test_features = feature_extractor.predict(X_test)

# Save features as .npy files
np.save('shufflenet_train_features.npy', train_features)
np.save('shufflenet_val_features.npy', val_features)
np.save('shufflenet_test_features.npy', test_features)

print("✅ ShuffleNetV2 features extracted and saved successfully!")


# GA implementation

# In[ ]:


with open('modified_alexnet_features_train.npy', 'rb') as f:
    content = f.read()
    print("File size in bytes:", len(content))


# In[ ]:


import numpy as np
import os

# Step 1: Load the raw binary file
file_path = 'modified_alexnet_features_train.npy'
with open(file_path, 'rb') as f:
    f.seek(128)  # Skip the .npy header (usually 128 bytes)
    raw_data = np.frombuffer(f.read(), dtype=np.float32)

# Step 2: Check total number of elements
print("Total elements:", raw_data.size)  # Should be 1310688

# Step 3: Try to reshape based on known image count (1330)
num_images = 1330
features_per_image = raw_data.size // num_images  # Should be 985
print("Features per image:", features_per_image)

# Step 4: Reshape and save
alexnet_features = raw_data[:num_images * features_per_image].reshape((num_images, features_per_image))
print("Recovered shape:", alexnet_features.shape)

# Step 5: Save the fixed version for future use
np.save('clean_alexnet_features_train.npy', alexnet_features)


# In[ ]:


alexnet_features = np.load('clean_alexnet_features_train.npy')


# In[ ]:


import numpy as np
import os

file_path = 'nasnet_train_features.npy'
file_size = os.path.getsize(file_path)  # bytes

# Try loading raw first
try:
    nasnet_raw = np.load(file_path, allow_pickle=True)
    print("Loaded shape:", nasnet_raw.shape)
    print("Dtype:", nasnet_raw.dtype)
except Exception as e:
    print("Error loading nasnet:", e)
    print("File size in bytes:", file_size)


# In[ ]:


import numpy as np
import os

# Step 1: Load the raw binary file for NASNet
file_path = 'nasnet_train_features.npy'
with open(file_path, 'rb') as f:
    f.seek(128)  # Skip the .npy header (usually 128 bytes)
    raw_data = np.frombuffer(f.read(), dtype=np.float32)

# Step 2: Check total number of elements
print("Total elements:", raw_data.size)  # Should be 262112

# Step 3: Try to reshape based on known image count (1330)
num_images = 1330
features_per_image = raw_data.size // num_images  # Should be 197
print("Features per image:", features_per_image)

# Step 4: Reshape and save
nasnet_features = raw_data[:num_images * features_per_image].reshape((num_images, features_per_image))
print("Recovered shape:", nasnet_features.shape)

# Step 5: Save the fixed version for future use
np.save('clean_nasnet_features_train.npy', nasnet_features)


# In[ ]:


# Load the reshaped NASNet features
nasnet_features = np.load('clean_nasnet_features_train.npy')
print(f"NASNet features shape: {nasnet_features.shape}")


# In[ ]:


with open('modified_alexnet_features_val.npy', 'rb') as f:
    content = f.read()
    print("File size in bytes:", len(content))


# In[ ]:


import numpy as np
import os

# Step 1: Load the raw binary file for validation
file_path_val = 'modified_alexnet_features_val.npy'
with open(file_path_val, 'rb') as f:
    f.seek(128)  # Skip the .npy header (usually 128 bytes)
    raw_data_val = np.frombuffer(f.read(), dtype=np.float32)

# Step 2: Check total number of elements
print("Total elements for validation:", raw_data_val.size)  # Check the total size

# Step 3: Try to reshape based on known image count (256)
num_images_val = 256
features_per_image_val = raw_data_val.size // num_images_val  # Calculate features per image
print("Features per image for validation:", features_per_image_val)

# Step 4: Reshape and save
alexnet_features_val = raw_data_val[:num_images_val * features_per_image_val].reshape((num_images_val, features_per_image_val))
print("Recovered shape for validation:", alexnet_features_val.shape)

# Step 5: Save the fixed version for future use
np.save('clean_alexnet_features_val.npy', alexnet_features_val)


# In[ ]:


# Load the reshaped AlexNet validation features
alexnet_features_val = np.load('clean_alexnet_features_val.npy')
print(f"AlexNet validation features shape: {alexnet_features_val.shape}")


# In[ ]:


import os
print("Current directory:", os.getcwd())


# In[ ]:


import os
print("Files in current directory:", os.listdir('/content'))


# In[ ]:


from sklearn.model_selection import train_test_split

# Split into 70% training and 30% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Save the labels
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
print("Saved 'y_train.npy' and 'y_val.npy'")


# In[ ]:


import numpy as np
import os

# Step 1: Load the raw binary file for ShuffleNet
file_path = 'shufflenet_train_features.npy'
with open(file_path, 'rb') as f:
    f.seek(128)  # Skip the .npy header (commonly around 128 bytes)
    raw_data = np.frombuffer(f.read(), dtype=np.float32)

# Step 2: Check total number of elements
print("Total elements:", raw_data.size)

# Step 3: Try to reshape based on known image count (1330)
num_images = 1330
features_per_image = raw_data.size // num_images
print("Features per image:", features_per_image)

# Step 4: Reshape and check
shufflenet_features = raw_data[:num_images * features_per_image].reshape((num_images, features_per_image))
print("Recovered shape:", shufflenet_features.shape)

# Step 5: Save the cleaned version
np.save('clean_shufflenet_features_train.npy', shufflenet_features)


# In[ ]:


import numpy as np
import os

# Step 1: Load the raw binary file for InceptionV3
file_path = 'inceptionv3_train_features.npy'
with open(file_path, 'rb') as f:
    f.seek(128)  # Skip the .npy header (usually around 128 bytes)
    raw_data = np.frombuffer(f.read(), dtype=np.float32)

# Step 2: Check total number of elements
print("Total elements:", raw_data.size)

# Step 3: Try to reshape based on known image count (1330)
num_images = 1330
features_per_image = raw_data.size // num_images
print("Features per image:", features_per_image)

# Step 4: Reshape and check
inceptionv3_features = raw_data[:num_images * features_per_image].reshape((num_images, features_per_image))
print("Recovered shape:", inceptionv3_features.shape)

# Step 5: Save the cleaned version
np.save('clean_inceptionv3_features_train.npy', inceptionv3_features)


# In[ ]:


import numpy as np

# Step 1: Load the training feature vectors for each model
shufflenet_train_features = np.load('clean_shufflenet_features_train.npy')
vgg19_train_features = np.load('vgg19_gap_train_features.npy')
inceptionv3_train_features = np.load('inceptionv3_features_train.npy')
alexnet_train_features = np.load('clean_alexnet_features_train.npy')
nasnet_train_features = np.load('clean_nasnet_features_train.npy')

# Step 2: Flatten the 4D arrays (if they exist) in training features
if len(shufflenet_train_features.shape) == 4:
    shufflenet_train_features = shufflenet_train_features.reshape(shufflenet_train_features.shape[0], -1)
if len(vgg19_train_features.shape) == 4:
    vgg19_train_features = vgg19_train_features.reshape(vgg19_train_features.shape[0], -1)
if len(inceptionv3_train_features.shape) == 4:
    inceptionv3_train_features = inceptionv3_train_features.reshape(inceptionv3_train_features.shape[0], -1)
if len(alexnet_train_features.shape) == 4:
    alexnet_train_features = alexnet_train_features.reshape(alexnet_train_features.shape[0], -1)
if len(nasnet_train_features.shape) == 4:
    nasnet_train_features = nasnet_train_features.reshape(nasnet_train_features.shape[0], -1)

# Step 3: Concatenate all the training feature vectors
X_train_features = np.concatenate([
    shufflenet_train_features,
    vgg19_train_features,
    inceptionv3_train_features,
    alexnet_train_features,
    nasnet_train_features
], axis=1)

# Step 4: Load the validation feature vectors for each model
shufflenet_val_features = np.load('shufflenet_val_features.npy')
vgg19_val_features = np.load('vgg19_gap_val_features.npy')
inceptionv3_val_features = np.load('inceptionv3_features_val.npy')
alexnet_val_features = np.load('clean_alexnet_features_val.npy')
nasnet_val_features = np.load('nasnet_val_features.npy')

# Step 5: Flatten the 4D arrays (if they exist) in validation features
if len(shufflenet_val_features.shape) == 4:
    shufflenet_val_features = shufflenet_val_features.reshape(shufflenet_val_features.shape[0], -1)
if len(vgg19_val_features.shape) == 4:
    vgg19_val_features = vgg19_val_features.reshape(vgg19_val_features.shape[0], -1)
if len(inceptionv3_val_features.shape) == 4:
    inceptionv3_val_features = inceptionv3_val_features.reshape(inceptionv3_val_features.shape[0], -1)
if len(alexnet_val_features.shape) == 4:
    alexnet_val_features = alexnet_val_features.reshape(alexnet_val_features.shape[0], -1)
if len(nasnet_val_features.shape) == 4:
    nasnet_val_features = nasnet_val_features.reshape(nasnet_val_features.shape[0], -1)

# Step 6: Check the shape of all validation feature arrays
print("shufflenet_val_features shape:", shufflenet_val_features.shape)
print("vgg19_val_features shape:", vgg19_val_features.shape)
print("inceptionv3_val_features shape:", inceptionv3_val_features.shape)
print("alexnet_val_features shape:", alexnet_val_features.shape)
print("nasnet_val_features shape:", nasnet_val_features.shape)

# Step 7: Ensure all validation sets have the same number of samples
min_samples = min(
    shufflenet_val_features.shape[0],
    vgg19_val_features.shape[0],
    inceptionv3_val_features.shape[0],
    alexnet_val_features.shape[0],
    nasnet_val_features.shape[0]
)

# Step 8: Trim the arrays to the minimum number of samples
shufflenet_val_features = shufflenet_val_features[:min_samples]
vgg19_val_features = vgg19_val_features[:min_samples]
inceptionv3_val_features = inceptionv3_val_features[:min_samples]
alexnet_val_features = alexnet_val_features[:min_samples]
nasnet_val_features = nasnet_val_features[:min_samples]

# Step 9: Concatenate validation features after trimming
X_val_features = np.concatenate([
    shufflenet_val_features,
    vgg19_val_features,
    inceptionv3_val_features,
    alexnet_val_features,
    nasnet_val_features
], axis=1)

# Step 10: Load the labels for both training and validation sets
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Step 11: Check the shape of the concatenated feature matrices
print("Shape of X_train_features:", X_train_features.shape)
print("Shape of X_val_features:", X_val_features.shape)


# In[ ]:


from tensorflow.keras import models, layers

def build_nn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

def evaluate_fitness(individual, X_train, y_train, X_val, y_val):
    selected_indices = np.where(individual == 1)[0]

    if len(selected_indices) == 0:
        return 0  # Avoid training with no features

    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]

    model = build_nn(X_train_selected.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    model.fit(X_train_selected, y_train,
              validation_data=(X_val_selected, y_val),
              epochs=10, batch_size=32, verbose=0, callbacks=[early_stop])

    _, val_acc = model.evaluate(X_val_selected, y_val, verbose=0)
    return val_acc


# In[ ]:


def initialize_population(pop_size, num_features):
    return np.random.randint(2, size=(pop_size, num_features))

def tournament_selection(pop, fitness_scores, k=3):
    selected = []
    for _ in range(len(pop)):
        contenders = np.random.choice(len(pop), k, replace=False)
        winner = pop[contenders[np.argmax(fitness_scores[contenders])]]
        selected.append(winner)
    return np.array(selected)

def single_point_crossover(parent1, parent2, rate=0.8):
    if np.random.rand() < rate:
        point = np.random.randint(1, len(parent1))
        return np.concatenate([parent1[:point], parent2[point:]]), \
               np.concatenate([parent2[:point], parent1[point:]])
    return parent1.copy(), parent2.copy()

def bit_flip_mutation(individual, rate=0.05):
    for i in range(len(individual)):
        if np.random.rand() < rate:
            individual[i] ^= 1  # Flip bit
    return individual


# In[ ]:


import os
import numpy as np
import pickle

def save_checkpoint(gen, population, best_fitness, best_solution, path="ga_checkpoint.pkl"):
    checkpoint = {
        "generation": gen,
        "population": population,
        "best_fitness": best_fitness,
        "best_solution": best_solution
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(path="ga_checkpoint.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def run_ga(X_train, y_train, X_val, y_val, generations=10, pop_size=20, checkpoint_path="ga_checkpoint.pkl"):
    num_features = X_train.shape[1]

    # Load checkpoint if available
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint:
        start_gen = checkpoint["generation"] + 1
        population = checkpoint["population"]
        best_fitness = checkpoint["best_fitness"]
        best_solution = checkpoint["best_solution"]
        print(f"Resuming from generation {start_gen}")
    else:
        start_gen = 1
        population = initialize_population(pop_size, num_features)
        best_fitness = -np.inf
        best_solution = None
        print("Starting new GA run")

    for gen in range(start_gen, generations + 1):
        print(f"\nGeneration {gen}/{generations}")
        fitness_scores = np.array([
            evaluate_fitness(ind, X_train, y_train, X_val, y_val) for ind in population
        ])

        best_gen_fit = np.max(fitness_scores)
        best_gen_sol = population[np.argmax(fitness_scores)]

        if best_gen_fit > best_fitness:
            best_fitness = best_gen_fit
            best_solution = best_gen_sol.copy()

        print(f"Best fitness in Gen {gen}: {best_gen_fit:.4f}")

        selected = tournament_selection(population, fitness_scores)
        next_gen = []

        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = single_point_crossover(parent1, parent2)
            next_gen.append(bit_flip_mutation(child1))
            next_gen.append(bit_flip_mutation(child2))

        population = np.array(next_gen)

        # Save checkpoint after each generation
        save_checkpoint(gen, population, best_fitness, best_solution, checkpoint_path)

    return best_solution


# In[ ]:


print("X_train_features shape:", X_train_features.shape)
print("y_train shape:", y_train.shape)
print("X_val_features shape:", X_val_features.shape)
print("y_val shape:", y_val.shape)


# In[ ]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# In[ ]:


y_val = y_val[:X_val_features.shape[0]]


# In[ ]:


# Run GA to get best feature mask
best_features = run_ga(X_train_features, y_train, X_val_features, y_val)

# Extract selected features
X_train_selected = X_train_features[:, best_features == 1]
X_val_selected = X_val_features[:, best_features == 1]

# Train final model
final_model = build_nn(X_train_selected.shape[1])
final_model.fit(X_train_selected, y_train, epochs=10, batch_size=32, verbose=1)

# Final evaluation
_, final_acc = final_model.evaluate(X_val_selected, y_val)
print("Final Validation Accuracy with GA-selected features:", final_acc)


# In[ ]:


import numpy as np

# Step 1: Load the training feature vectors for each model

alexnet_train_features = np.load('clean_alexnet_features_train.npy')


# Step 2: Flatten the 4D arrays (if they exist) in training features

if len(alexnet_train_features.shape) == 4:
    alexnet_train_features = alexnet_train_features.reshape(alexnet_train_features.shape[0], -1)


# Step 3: Concatenate all the training feature vectors
X_train_features = np.concatenate([
    alexnet_train_features
], axis=1)

# Step 4: Load the validation feature vectors for each model
alexnet_val_features = np.load('clean_alexnet_features_val.npy')


# Step 5: Flatten the 4D arrays (if they exist) in validation features

if len(alexnet_val_features.shape) == 4:
    alexnet_val_features = alexnet_val_features.reshape(alexnet_val_features.shape[0], -1)

# Step 6: Check the shape of all validation feature arrays

print("alexnet_val_features shape:", alexnet_val_features.shape)




# Step 9: Concatenate validation features after trimming
X_val_features = np.concatenate([

    alexnet_val_features

], axis=1)

# Step 10: Load the labels for both training and validation sets
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Step 11: Check the shape of the concatenated feature matrices
print("Shape of X_train_features:", X_train_features.shape)
print("Shape of X_val_features:", X_val_features.shape)


# In[ ]:


from tensorflow.keras import models, layers

def build_nn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

def evaluate_fitness(individual, X_train, y_train, X_val, y_val):
    selected_indices = np.where(individual == 1)[0]

    if len(selected_indices) == 0:
        return 0  # Avoid training with no features

    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]

    model = build_nn(X_train_selected.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    model.fit(X_train_selected, y_train,
              validation_data=(X_val_selected, y_val),
              epochs=10, batch_size=32, verbose=0, callbacks=[early_stop])

    _, val_acc = model.evaluate(X_val_selected, y_val, verbose=0)
    return val_acc


# In[ ]:


def initialize_population(pop_size, num_features):
    return np.random.randint(2, size=(pop_size, num_features))

def tournament_selection(pop, fitness_scores, k=3):
    selected = []
    for _ in range(len(pop)):
        contenders = np.random.choice(len(pop), k, replace=False)
        winner = pop[contenders[np.argmax(fitness_scores[contenders])]]
        selected.append(winner)
    return np.array(selected)

def single_point_crossover(parent1, parent2, rate=0.8):
    if np.random.rand() < rate:
        point = np.random.randint(1, len(parent1))
        return np.concatenate([parent1[:point], parent2[point:]]), \
               np.concatenate([parent2[:point], parent1[point:]])
    return parent1.copy(), parent2.copy()

def bit_flip_mutation(individual, rate=0.05):
    for i in range(len(individual)):
        if np.random.rand() < rate:
            individual[i] ^= 1  # Flip bit
    return individual


# In[ ]:


def run_ga(X_train, y_train, X_val, y_val, generations=10, pop_size=20):
    num_features = X_train.shape[1]
    population = initialize_population(pop_size, num_features)
    best_fitness = -np.inf
    best_solution = None

    for gen in range(generations):
        print(f"\nGeneration {gen+1}/{generations}")
        fitness_scores = np.array([
            evaluate_fitness(ind, X_train, y_train, X_val, y_val) for ind in population
        ])

        best_gen_fit = np.max(fitness_scores)
        best_gen_sol = population[np.argmax(fitness_scores)]

        if best_gen_fit > best_fitness:
            best_fitness = best_gen_fit
            best_solution = best_gen_sol.copy()

        print(f"Best fitness in Gen {gen+1}: {best_gen_fit:.4f}")

        selected = tournament_selection(population, fitness_scores)
        next_gen = []

        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1, child2 = single_point_crossover(parent1, parent2)
            next_gen.append(bit_flip_mutation(child1))
            next_gen.append(bit_flip_mutation(child2))

        population = np.array(next_gen)

    return best_solution


# In[ ]:


print("X_train_features.shape:", X_train_features.shape)
print("X_val_features.shape:", X_val_features.shape)
print("best_features.shape:", best_features.shape)


# In[ ]:


# Rebuild X_val_features to match X_train_features
alexnet_val_features = np.load('clean_alexnet_features_val.npy')

if len(alexnet_val_features.shape) == 4:
    alexnet_val_features = alexnet_val_features.reshape(alexnet_val_features.shape[0], -1)

X_val_features = np.concatenate([alexnet_val_features], axis=1)  # Only AlexNet


# In[ ]:


print(X_train_features.shape)
print(X_val_features.shape)


# In[ ]:


# Step 4: Load the validation feature vectors for AlexNet only
alexnet_val_features = np.load('clean_alexnet_features_val.npy')

# Step 5: Flatten the 4D arrays (if they exist) in validation features
if len(alexnet_val_features.shape) == 4:
    alexnet_val_features = alexnet_val_features.reshape(alexnet_val_features.shape[0], -1)

# Step 6 (updated): Only use AlexNet features for validation
X_val_features = np.concatenate([
    alexnet_val_features
], axis=1)


# In[ ]:


print("X_train_features.shape:", X_train_features.shape)
print("X_val_features.shape:", X_val_features.shape)


# In[ ]:


import numpy as np

# Load AlexNet training and validation features
alexnet_train_features = np.load('clean_alexnet_features_train.npy')
alexnet_val_features = np.load('clean_alexnet_features_val.npy')

# Flatten if 4D
if alexnet_train_features.ndim == 4:
    alexnet_train_features = alexnet_train_features.reshape(alexnet_train_features.shape[0], -1)

if alexnet_val_features.ndim == 4:
    alexnet_val_features = alexnet_val_features.reshape(alexnet_val_features.shape[0], -1)

# Concatenate (only AlexNet used here)
X_train_features = np.concatenate([alexnet_train_features], axis=1)
X_val_features = np.concatenate([alexnet_val_features], axis=1)

# Load labels
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Check shapes
print("X_train_features.shape:", X_train_features.shape)
print("X_val_features.shape:", X_val_features.shape)


# In[ ]:


# Load the raw .npy file again and print shape directly
alexnet_val_features = np.load('clean_alexnet_features_val.npy')
print("Raw alexnet_val_features.shape:", alexnet_val_features.shape)

# If 4D, flatten
if alexnet_val_features.ndim == 4:
    alexnet_val_features = alexnet_val_features.reshape(alexnet_val_features.shape[0], -1)

print("Flattened alexnet_val_features.shape:", alexnet_val_features.shape)

# Confirm that this is the only feature used in X_val_features
X_val_features = np.concatenate([alexnet_val_features], axis=1)
print("X_val_features.shape:", X_val_features.shape)


# In[ ]:


# Double-check the content of the list passed to np.concatenate
feature_list = [
    alexnet_val_features,

]

for i, feat in enumerate(feature_list):
    print(f"Feature {i} shape:", feat.shape)

X_val_features = np.concatenate(feature_list, axis=1)
print("Final X_val_features shape:", X_val_features.shape)


# In[ ]:


file_path_val = 'modified_alexnet_features_val.npy'
with open(file_path_val, 'rb') as f:
    f.seek(128)  # Skip the .npy header
    raw_data_val = np.frombuffer(f.read(), dtype=np.float32)

# Suppose 256 validation images
num_val_images = 256
features_per_image_val = raw_data_val.size // num_val_images
print("Validation features per image:", features_per_image_val)

# Reshape and slice to 4095 features
val_features = raw_data_val[:num_val_images * features_per_image_val].reshape((num_val_images, features_per_image_val))
val_features = val_features[:, :4095]

print("Final val shape:", val_features.shape)

np.save('clean_alexnet_features_val_985.npy', val_features)


# In[ ]:


y_val = y_val[:X_val_features.shape[0]]


# In[ ]:


# Reshape or truncate both training and validation features to have the same number of features
X_val_features = X_val_features[:, :X_train_features.shape[1]]  # Truncate to match the training features' size


# In[ ]:


# Run GA to get best feature mask
best_features = run_ga(X_train_features, y_train, X_val_features, y_val)

# Extract selected features
X_train_selected = X_train_features[:, best_features == 1]
X_val_selected = X_val_features[:, best_features == 1]

# Train final model
final_model = build_nn(X_train_selected.shape[1])
final_model.fit(X_train_selected, y_train, epochs=10, batch_size=32, verbose=1)

# Final evaluation
_, final_acc = final_model.evaluate(X_val_selected, y_val)
print("Final Validation Accuracy with GA-selected features:", final_acc)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_val_features = scaler.transform(X_val_features)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm(input_dim, num_classes):
    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=(1, input_dim), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


# Build the model
lstm_model = build_lstm(X_train_selected.shape[1], y_train_cat.shape[1])

# Train
lstm_model.fit(X_train_lstm, y_train_cat, epochs=50, batch_size=32, validation_data=(X_val_lstm, y_val_cat), verbose=1)

# Evaluate
_, final_acc = lstm_model.evaluate(X_val_lstm, y_val_cat)
print("Final Validation Accuracy with GA + LSTM:", final_acc)


# In[ ]:


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# === 1. Data Preparation ===
scaler = StandardScaler()
X_ga_train_scaled = scaler.fit_transform(X_ga_train[:, selected_features])
X_ga_val_scaled = scaler.transform(X_ga_val[:, selected_features])

# Ensure reshaping is correct: (samples, timesteps, features)
X_train = X_ga_train_scaled.reshape(X_ga_train_scaled.shape[0], 1, X_ga_train_scaled.shape[1])  # Change timesteps as needed
X_val = X_ga_val_scaled.reshape(X_ga_val_scaled.shape[0], 1, X_ga_val_scaled.shape[1])  # Change timesteps as needed

y_labels = np.argmax(y_ga_train, axis=1)
weights = class_weight.compute_class_weight(class_weight='balanced',
                                             classes=np.unique(y_labels),
                                             y=y_labels)
class_weights = dict(enumerate(weights))

# === 2. BiLSTM Model Architecture ===
def build_bilstm_model(input_shape):
    inputs = Input(shape=input_shape)

    # Bidirectional LSTM Layer
    x = Bidirectional(LSTM(128,  # Increased units to allow for better feature learning
                           return_sequences=False,
                           kernel_regularizer=l2(0.01),
                           recurrent_regularizer=l2(0.01)))(inputs)
    x = Dropout(0.3)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalize activations to speed up convergence

    # Dense Layer
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)

    # Output Layer
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_bilstm_model((1, X_train.shape[2]))

# === 3. Model Compilation ===
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipvalue=0.5
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# === 4. Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ModelCheckpoint('best_bilstm_model.keras', save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)
]

# === 5. Model Training ===
history = model.fit(
    X_train, y_ga_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_ga_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# === 6. Evaluation ===
val_probs = model.predict(X_val)
val_preds = np.argmax(val_probs, axis=1)
y_true = np.argmax(y_ga_val, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, val_preds, digits=4))

accuracy = accuracy_score(y_true, val_preds)
fpr, tpr, _ = roc_curve(y_true, val_probs[:, 1])
roc_auc = auc(fpr, tpr)

print(f"\nFinal Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")

# === 7. ROC Curve ===
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()



# ===== File: miniproject-brainvista.py =====
#!/usr/bin/env python
# coding: utf-8

# In[1]:


normal_path = "/kaggle/input/brains/Brain_Data_Organised/Normal"
stroke_path = "/kaggle/input/brains/Brain_Data_Organised/Stroke"


# In[2]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.nasnet import NASNetLarge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[3]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, VGG19, NASNetLarge
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_nasnet
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D

# Define paths
normal_path = "/kaggle/input/brains/Brain_Data_Organised/Normal"
stroke_path = "/kaggle/input/brains/Brain_Data_Organised/Stroke"

# Load and preprocess images
def load_and_preprocess_images(normal_path, stroke_path, target_size=(227, 227)):
    all_images = []
    all_labels = []

    for img_name in os.listdir(normal_path):
        img_path = os.path.join(normal_path, img_name)
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        all_images.append(img)
        all_labels.append(0)  # Normal

    for img_name in os.listdir(stroke_path):
        img_path = os.path.join(stroke_path, img_name)
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        all_images.append(img)
        all_labels.append(1)  # Stroke

    return np.array(all_images), np.array(all_labels)

# Load images
all_images, all_labels = load_and_preprocess_images(normal_path, stroke_path)

print(f"Shape of all_images: {all_images.shape}")
print(f"Shape of all_labels: {all_labels.shape}")

# One-hot encode labels
encoded_labels = to_categorical(all_labels)
print("Encoded labels (first 10):", encoded_labels[:10])

# Train-test-validation split
X_train, X_test, y_train, y_test = train_test_split(all_images, encoded_labels, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# AlexNet custom model
def create_alexnet(input_shape=(227, 227, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(96, (11, 11), strides=4, activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

# Simplified ShuffleNet custom model
def create_shufflenet(input_shape=(227, 227, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(24, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = Conv2D(144, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(144, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(288, (1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=inputs, outputs=x)
    return model

# Initialize models
alexnet_model = create_alexnet()
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
vgg19_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
nasnet_model = NASNetLarge(weights='imagenet', include_top=False, pooling='avg', input_shape=(331, 331, 3))
shufflenet_model = create_shufflenet()

# Preprocessing function
def preprocess_for_model(images, model_name):
    if model_name == 'inception':
        return preprocess_inception(images.copy())
    elif model_name == 'vgg19':
        return preprocess_vgg19(images.copy())
    elif model_name == 'nasnet':
        resized = tf.image.resize(images, (331, 331))
        return preprocess_nasnet(resized.numpy())
    elif model_name in ['alexnet', 'shufflenet']:
        return (images / 255.0).astype('float32')
    else:
        return images

# Feature extraction
def extract_features(model, images, model_name, batch_size=32):
    processed_images = preprocess_for_model(images, model_name)
    return model.predict(processed_images, batch_size=batch_size, verbose=1)

# Extract and print features
print("\nExtracting features from AlexNet...")
alexnet_features = extract_features(alexnet_model, X_train, 'alexnet')

print("\nExtracting features from InceptionV3...")
inception_features = extract_features(inception_model, X_train, 'inception')

print("\nExtracting features from VGG19...")
vgg19_features = extract_features(vgg19_model, X_train, 'vgg19')

print("\nExtracting features from NASNetLarge...")
nasnet_features = extract_features(nasnet_model, X_train, 'nasnet')

print("\nExtracting features from ShuffleNet...")
shufflenet_features = extract_features(shufflenet_model, X_train, 'shufflenet')

# Print feature shapes
print(f"\nShape of AlexNet features: {alexnet_features.shape}")
print(f"Shape of InceptionV3 features: {inception_features.shape}")
print(f"Shape of VGG19 features: {vgg19_features.shape}")
print(f"Shape of NASNetLarge features: {nasnet_features.shape}")
print(f"Shape of ShuffleNet features: {shufflenet_features.shape}")

# Combine features
combined_features = np.concatenate([
    alexnet_features,
    inception_features,
    vgg19_features,
    nasnet_features,
    shufflenet_features
], axis=1)

print(f"\nShape of combined features: {combined_features.shape}")


# In[ ]:





# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import os

# Optional: for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Assume combined_features and y_train from your previous code
X_ga_train, X_ga_val, y_ga_train, y_ga_val = train_test_split(
    combined_features, y_train, test_size=0.2, random_state=seed
)

# GA parameters
num_features = combined_features.shape[1]
population_size = 20
num_generations = 10
crossover_rate = 0.8
mutation_rate = 0.02
num_parents = 10

# Initialize population (binary chromosomes)
def initialize_population():
    return np.random.randint(0, 2, size=(population_size, num_features))

# Fitness function using a small NN
def fitness(chromosome):
    selected_indices = np.where(chromosome == 1)[0]
    if len(selected_indices) == 0:
        return 0  # avoid empty feature sets

    X_sel_train = X_ga_train[:, selected_indices]
    X_sel_val = X_ga_val[:, selected_indices]

    model = Sequential([
        Input(shape=(X_sel_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_sel_train, y_ga_train, epochs=10, batch_size=32, verbose=0)

    val_preds = model.predict(X_sel_val, verbose=0)
    acc = accuracy_score(np.argmax(y_ga_val, axis=1), np.argmax(val_preds, axis=1))
    return acc

# Select top-k parents
def select_parents(population, fitness_scores, k):
    idx = np.argsort(fitness_scores)[-k:]
    return population[idx]

# Single-point crossover
def crossover(parents):
    offspring = []
    while len(offspring) < population_size - len(parents):
        parent1, parent2 = random.sample(list(parents), 2)
        if np.random.rand() < crossover_rate:
            point = random.randint(1, num_features - 2)
            child = np.concatenate([parent1[:point], parent2[point:]])
        else:
            child = parent1.copy()
        offspring.append(child)
    return np.array(offspring)

# Mutation (bit flip)
def mutate(offspring):
    for child in offspring:
        for i in range(num_features):
            if np.random.rand() < mutation_rate:
                child[i] = 1 - child[i]
    return offspring

# Run GA
population = initialize_population()

for gen in range(num_generations):
    print(f"\n Generation {gen+1}/{num_generations}")

    fitness_scores = []
    for i, chromo in enumerate(population):
        acc = fitness(chromo)
        fitness_scores.append(acc)
        print(f"Chromosome {i+1}/{population_size} → Accuracy: {acc:.4f}")

    best_idx = np.argmax(fitness_scores)
    print(f" Best Accuracy this Gen: {fitness_scores[best_idx]:.4f}")

    parents = select_parents(population, fitness_scores, num_parents)
    offspring = crossover(parents)
    offspring = mutate(offspring)
    population = np.vstack((parents, offspring))

# Final best solution
final_fitness_scores = [fitness(chromo) for chromo in population]
best_chromosome = population[np.argmax(final_fitness_scores)]
selected_features = np.where(best_chromosome == 1)[0]

print(f"\n Total selected features: {len(selected_features)}")
print(f" Selected feature indices: {selected_features}")


# GA_LSTM

# In[5]:


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# === 1. Data Preparation ===
# Scale features
scaler = StandardScaler()
X_ga_train_scaled = scaler.fit_transform(X_ga_train[:, selected_features])
X_ga_val_scaled = scaler.transform(X_ga_val[:, selected_features])

# Reshape for LSTM [samples, timesteps=1, features]
X_train = X_ga_train_scaled.reshape(X_ga_train_scaled.shape[0], 1, X_ga_train_scaled.shape[1])
X_val = X_ga_val_scaled.reshape(X_ga_val_scaled.shape[0], 1, X_ga_val_scaled.shape[1])

# Class weights for imbalance
y_labels = np.argmax(y_ga_train, axis=1)
weights = class_weight.compute_class_weight(class_weight='balanced', 
                                          classes=np.unique(y_labels), 
                                          y=y_labels)
class_weights = dict(enumerate(weights))

# === 2. LSTM Model Architecture (Matching Paper) ===
def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)

    # LSTM Layer with 64 units (as in paper)
    x = LSTM(64, 
             return_sequences=False,
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01))(inputs)
    x = Dropout(0.3)(x)  # 30% dropout as in paper
    x = BatchNormalization()(x)

    # Dense Layer with 64 units (as in paper)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)  # 20% dropout as in paper

    # Output layer
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_lstm_model((1, X_train.shape[2]))

# === 3. Model Compilation ===
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipvalue=0.5  # Gradient clipping
)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# === 4. Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_lstm_model.keras', save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)
]

# === 5. Model Training ===
history = model.fit(
    X_train, y_ga_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_ga_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# === 6. Evaluation ===
val_probs = model.predict(X_val)
val_preds = np.argmax(val_probs, axis=1)
y_true = np.argmax(y_ga_val, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, val_preds, digits=4))

# Calculate metrics
accuracy = accuracy_score(y_true, val_preds)
fpr, tpr, _ = roc_curve(y_true, val_probs[:, 1])
roc_auc = auc(fpr, tpr)

print(f"\nFinal Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")

# === 7. ROC Curve ===
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# === 8. Learning Curves ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# GA+BILSTM

# In[6]:


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# === 1. Data Preparation ===
scaler = StandardScaler()
X_ga_train_scaled = scaler.fit_transform(X_ga_train[:, selected_features])
X_ga_val_scaled = scaler.transform(X_ga_val[:, selected_features])

# Reshape for LSTM [samples, timesteps=1, features]
X_train = X_ga_train_scaled.reshape(X_ga_train_scaled.shape[0], 1, X_ga_train_scaled.shape[1])
X_val = X_ga_val_scaled.reshape(X_ga_val_scaled.shape[0], 1, X_ga_val_scaled.shape[1])

# Compute class weights
y_labels = np.argmax(y_ga_train, axis=1)
weights = class_weight.compute_class_weight(class_weight='balanced',
                                             classes=np.unique(y_labels),
                                             y=y_labels)
class_weights = dict(enumerate(weights))

# === 2. BiLSTM Model Architecture ===
def build_bilstm_model(input_shape):
    inputs = Input(shape=input_shape)

    # BiLSTM Layer with regularization
    x = Bidirectional(LSTM(64,
                           return_sequences=False,
                           kernel_regularizer=l2(0.01),
                           recurrent_regularizer=l2(0.01)))(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    # Dense layer
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)

    # Output layer
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_bilstm_model((1, X_train.shape[2]))

# === 3. Compile Model ===
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

# === 4. Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
    ModelCheckpoint('best_bilstm_model.keras', save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=6, min_lr=1e-6)
]

# === 5. Train Model ===
history = model.fit(
    X_train, y_ga_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_val, y_ga_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# === 6. Evaluation ===
val_probs = model.predict(X_val)
val_preds = np.argmax(val_probs, axis=1)
y_true = np.argmax(y_ga_val, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, val_preds, digits=4))

accuracy = accuracy_score(y_true, val_preds)
fpr, tpr, _ = roc_curve(y_true, val_probs[:, 1])
roc_auc = auc(fpr, tpr)

print(f"\nFinal Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")

# === 7. ROC Curve ===
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# === 8. Learning Curves ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()




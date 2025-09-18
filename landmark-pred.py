import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

SEED = 42
IMG_SIZE = 128
DATASET_PATH = '../Input/landmark-recognition-2021/'
MIN_CLASS = 0
MAX_CLASS = 500
UNDERSAMPLING_THRESHOLD = 80
OVERSAMPLING_FACTOR = 0.2
KERNEL_SIZE = (3, 3)
PADDING_TYPE = 'same'
ACTIVATION_FUNC = 'relu'
POOL_SIZE = (2, 2)
STRIDES_SIZE = (2, 2)
DROPOUT_RATE = 0.5
BATCH_SIZE = 256
EPOCHS = 1000
LEARNING_RATE = 0.01
WEIGHT_DECAY = tf.keras.regularizers.L2(0.01)

def load_traindf(base_path):
    """Loads and prepares the training dataframe."""
    csv_path = os.path.join(base_path, 'train.csv')
    train_dir = os.path.join(base_path, 'train')
    
    print(f"Loading data from {csv_path}...")
    traindf = pd.read_csv(csv_path)
    traindf['img_path'] = traindf['id'].apply(
        lambda r: os.path.join(train_dir, r[0], r[1], r[2], r + '.jpg')
    )
    traindf['landmark_id'] = traindf['landmark_id'].astype(np.int32)
    return traindf

def img_read_resize(img_path):
    img = plt.imread(img_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img_resized

def create_model(n_class, train_data, train_labels, val_data, val_labels, class_weights):

    opt1 = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9, momentum=0.5, epsilon=1e-07)
    opt2 = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    opt3 = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.2, 0.2)),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.4),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.1, 0.1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=KERNEL_SIZE, strides=1, padding=PADDING_TYPE, activation=ACTIVATION_FUNC),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES_SIZE, padding=PADDING_TYPE),
        tf.keras.layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=1, padding=PADDING_TYPE, activation=ACTIVATION_FUNC),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES_SIZE, padding=PADDING_TYPE),
        tf.keras.layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, padding=PADDING_TYPE, activation=ACTIVATION_FUNC),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES_SIZE, padding=PADDING_TYPE),
        tf.keras.layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, padding=PADDING_TYPE, activation=ACTIVATION_FUNC),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES_SIZE, padding=PADDING_TYPE),
        tf.keras.layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, padding=PADDING_TYPE, activation=ACTIVATION_FUNC),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES_SIZE, padding=PADDING_TYPE),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation=ACTIVATION_FUNC, kernel_regularizer=WEIGHT_DECAY),
        tf.keras.layers.Dropout(rate=DROPOUT_RATE, seed=SEED),
        tf.keras.layers.Dense(units=n_class, activation='softmax')
    ])
    model.compile(
        optimizer=opt3,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0.0001, patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, min_delta=0.001, cooldown=1, min_lr=0.0005, verbose=1)
    checkpoint = ModelCheckpoint('best-weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
    print("\nStarting model training...")
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        class_weight=class_weights,
        shuffle=True,
        batch_size=BATCH_SIZE,
        steps_per_epoch=len(train_data) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1,
        # use_multiprocessing=True,
        # workers=32,
    )
    return model, history

if __name__ == "__main__":

    traindf = load_traindf(DATASET_PATH)
    print(f"\nSubsetting data to classes between {MIN_CLASS} and {MAX_CLASS}...")
    traindf_s = traindf[(traindf['landmark_id'] > MIN_CLASS) & (traindf['landmark_id'] < MAX_CLASS)]  
    print(f"Undersampling classes to a max of {UNDERSAMPLING_THRESHOLD} images per class...")
    traindf_s = (traindf_s.groupby('landmark_id', group_keys=False)
                 .apply(lambda x: x.sample(n=min(len(x), UNDERSAMPLING_THRESHOLD), random_state=SEED)))
    traindf_s.reset_index(inplace=True, drop=True)
    N_CLASS = len(traindf_s['landmark_id'].unique())
    print(f"Total images after undersampling: {len(traindf_s)}")
    print(f"Total unique classes after undersampling: {N_CLASS}\n")
    print("Loading images into memory...")
    X = [img_read_resize(path) for path in traindf_s['img_path']]
    y = traindf_s['landmark_id'].values
    print("Performing label encoding and image normalization...")
    LE = LabelEncoder()
    y_LE = LE.fit_transform(y)
    X = np.array(X) / 255.0
    print("Splitting data into training, validation, and test sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_LE, test_size=0.10, random_state=SEED, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.05, random_state=SEED, shuffle=True)
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}\n")
    print("Applying oversampling to minority classes in the training set...")
    OVERSAMPLING_THRESHOLD = int(OVERSAMPLING_FACTOR * UNDERSAMPLING_THRESHOLD)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    
    for cls, count in zip(unique_classes, class_counts):
        if count < OVERSAMPLING_THRESHOLD:
            img_class_i = X_train[y_train == cls]
            samples_to_add = OVERSAMPLING_THRESHOLD - count
            random_indices = np.random.choice(len(img_class_i), size=samples_to_add, replace=True)
            X_train = np.concatenate((X_train, img_class_i[random_indices]), axis=0)
            y_train = np.concatenate((y_train, np.full(shape=samples_to_add, fill_value=cls)))
    print(f"Training set size after oversampling: {len(X_train)}\n")

    print("Calculating class weights for model training...")
    classWeights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    train_classWeights = dict(enumerate(classWeights))
    freq, _ = np.unique(y_train, return_counts=True)
    print(f"Highest frequency: {np.max(freq)}")

    model, history = create_model(N_CLASS, X_train, y_train, X_val, y_val, train_classWeights)
    
    print("\nPlotting training and validation loss...")
    history_df = pd.DataFrame(history.history)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    print("\nEvaluation of the model on test data")
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}\n")

    print("Making predictions on the test set...")
    predict = model.predict(X_test, verbose=1)
    
    y_pred = []
    confidence = []
    for i in range(len(predict)):
        pred_class = np.argmax(predict[i])
        y_pred.append(pred_class)
        confidence.append(predict[i][pred_class].round(2))
        
    y_pred_labels = LE.inverse_transform(y_pred)
    y_test_labels = LE.inverse_transform(y_test)

    print("\nSample Predictions (Predicted Label | True Label | Confidence):")
    for i in range(min(10, len(y_pred_labels))):
        print(f"{y_pred_labels[i]:<15} | {y_test_labels[i]:<15} | {confidence[i]}")
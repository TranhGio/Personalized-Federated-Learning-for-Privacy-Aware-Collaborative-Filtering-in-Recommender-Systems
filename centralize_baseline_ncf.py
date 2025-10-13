"""
Centralized Baseline: Neural Collaborative Filtering (NCF) on MovieLens-1M
Using Keras/TensorFlow
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as sklearn_split
from datetime import datetime
import os

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class Config:
    RESULTS_DIR = 'results/centralized'
    FIGURES_DIR = 'figures'
    EMBEDDING_DIM = 50
    HIDDEN_LAYERS = [128, 64, 32]
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 0.001

    def __init__(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.FIGURES_DIR, exist_ok=True)


config = Config()


# ============================================
# DATA PREPARATION
# ============================================

def load_and_prepare_data():
    """
    Load MovieLens-1M and prepare for NCF
    Returns: train and test data
    """
    print("=" * 60)
    print("LOADING AND PREPARING DATA FOR NCF")
    print("=" * 60)

    # Download and load data
    from surprise import Dataset
    data = Dataset.load_builtin('ml-1m')

    # Convert to DataFrame
    raw_ratings = data.raw_ratings
    df = pd.DataFrame(raw_ratings, columns=['user_id', 'item_id', 'rating', 'timestamp'])

    # Create user and item mappings (0-indexed)
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    df['user_idx'] = df['user_id'].map(user2idx)
    df['item_idx'] = df['item_id'].map(item2idx)

    n_users = len(user_ids)
    n_items = len(item_ids)

    print(f"✓ Data prepared")
    print(f"  - Users: {n_users:,}")
    print(f"  - Items: {n_items:,}")
    print(f"  - Ratings: {len(df):,}")

    # Split data
    train_df, test_df = sklearn_split(df, test_size=0.2, random_state=42)

    print(f"  - Train: {len(train_df):,}")
    print(f"  - Test: {len(test_df):,}\n")

    return train_df, test_df, n_users, n_items


# ============================================
# NCF MODEL
# ============================================

def build_ncf_model(n_users, n_items, embedding_dim=50, hidden_layers=[128, 64, 32]):
    """
    Build Neural Collaborative Filtering model

    Architecture:
    - GMF (Generalized Matrix Factorization) path
    - MLP (Multi-Layer Perceptron) path
    - NeuMF (Neural Matrix Factorization) - combination of GMF and MLP

    Args:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Embedding dimension
        hidden_layers: List of hidden layer sizes

    Returns:
        model: Compiled Keras model
    """

    # Input layers
    user_input = layers.Input(shape=(1,), name='user_input')
    item_input = layers.Input(shape=(1,), name='item_input')

    # ========== GMF Path ==========
    # User embedding for GMF
    gmf_user_embedding = layers.Embedding(
        n_users, embedding_dim,
        embeddings_initializer='he_normal',
        name='gmf_user_embedding'
    )(user_input)
    gmf_user_vec = layers.Flatten()(gmf_user_embedding)

    # Item embedding for GMF
    gmf_item_embedding = layers.Embedding(
        n_items, embedding_dim,
        embeddings_initializer='he_normal',
        name='gmf_item_embedding'
    )(item_input)
    gmf_item_vec = layers.Flatten()(gmf_item_embedding)

    # Element-wise product
    gmf_vector = layers.Multiply()([gmf_user_vec, gmf_item_vec])

    # ========== MLP Path ==========
    # User embedding for MLP
    mlp_user_embedding = layers.Embedding(
        n_users, embedding_dim,
        embeddings_initializer='he_normal',
        name='mlp_user_embedding'
    )(user_input)
    mlp_user_vec = layers.Flatten()(mlp_user_embedding)

    # Item embedding for MLP
    mlp_item_embedding = layers.Embedding(
        n_items, embedding_dim,
        embeddings_initializer='he_normal',
        name='mlp_item_embedding'
    )(item_input)
    mlp_item_vec = layers.Flatten()(mlp_item_embedding)

    # Concatenate user and item vectors
    mlp_vector = layers.Concatenate()([mlp_user_vec, mlp_item_vec])

    # MLP layers
    for hidden_size in hidden_layers:
        mlp_vector = layers.Dense(
            hidden_size,
            activation='relu',
            kernel_initializer='he_normal',
            name=f'mlp_dense_{hidden_size}'
        )(mlp_vector)
        mlp_vector = layers.BatchNormalization()(mlp_vector)
        mlp_vector = layers.Dropout(0.2)(mlp_vector)

    # ========== NeuMF ==========
    # Concatenate GMF and MLP paths
    neumf_vector = layers.Concatenate()([gmf_vector, mlp_vector])

    # Output layer
    output = layers.Dense(
        1,
        activation='linear',
        kernel_initializer='lecun_normal',
        name='prediction'
    )(neumf_vector)

    # Build model
    model = Model(
        inputs=[user_input, item_input],
        outputs=output,
        name='NeuMF'
    )

    return model


def train_ncf_model(model, train_df, test_df, epochs=20, batch_size=256):
    """
    Train NCF model

    Args:
        model: Compiled Keras model
        train_df: Training DataFrame
        test_df: Test DataFrame
        epochs: Number of epochs
        batch_size: Batch size

    Returns:
        history: Training history
    """
    print("=" * 60)
    print("TRAINING NCF MODEL")
    print("=" * 60)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    print(model.summary())

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Prepare data
    X_train = [train_df['user_idx'].values, train_df['item_idx'].values]
    y_train = train_df['rating'].values

    X_test = [test_df['user_idx'].values, test_df['item_idx'].values]
    y_test = test_df['rating'].values

    # Train
    print("\nTraining...")
    start_time = datetime.now()

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    training_time = (datetime.now() - start_time).total_seconds()

    print(f"\n✓ Training completed in {training_time:.2f} seconds")

    return history, training_time


def evaluate_ncf(model, test_df):
    """
    Evaluate NCF model

    Args:
        model: Trained Keras model
        test_df: Test DataFrame

    Returns:
        results: Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING NCF MODEL")
    print("=" * 60)

    X_test = [test_df['user_idx'].values, test_df['item_idx'].values]
    y_test = test_df['rating'].values

    # Predictions
    y_pred = model.predict(X_test, batch_size=1024, verbose=1).flatten()

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))

    # R² score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    results = {
        'model_name': 'NCF',
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2)
    }

    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"R²:   {r2:.4f}")
    print("=" * 60 + "\n")

    return results, y_pred


# ============================================
# VISUALIZATION
# ============================================

def plot_ncf_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training History - Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training History - MAE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")

    plt.show()


# ============================================
# MAIN
# ============================================

def main():
    """Main execution for NCF"""

    print("\n" + "=" * 60)
    print("CENTRALIZED BASELINE: NCF ON MOVIELENS-1M")
    print("=" * 60 + "\n")

    # 1. Load data
    train_df, test_df, n_users, n_items = load_and_prepare_data()

    # 2. Build model
    print("Building NCF model...")
    model = build_ncf_model(
        n_users,
        n_items,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_layers=config.HIDDEN_LAYERS
    )

    # 3. Train model
    history, training_time = train_ncf_model(
        model,
        train_df,
        test_df,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE
    )

    # 4. Evaluate
    results, predictions = evaluate_ncf(model, test_df)
    results['training_time'] = training_time

    # 5. Visualize
    plot_ncf_training_history(
        history,
        save_path=os.path.join(config.FIGURES_DIR, 'ncf_training_history.png')
    )

    # 6. Save model
    model_path = os.path.join(config.RESULTS_DIR, 'ncf_model.keras')
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")

    # 7. Save results
    import json
    results_path = os.path.join(config.RESULTS_DIR, 'ncf_baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results saved to {results_path}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY - NCF")
    print("=" * 60)
    print(f"RMSE:          {results['rmse']:.4f}")
    print(f"MAE:           {results['mae']:.4f}")
    print(f"R²:            {results['r2']:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print("=" * 60 + "\n")

    return model, results


if __name__ == "__main__":
    ncf_model, ncf_results = main()
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from google.cloud import bigquery_storage
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Check if CUDA is available
device = "cuda"

# Load data from BigQuery using BigQuery Storage API
print("Loading data from BigQuery...")
client = bigquery_storage.BigQueryReadClient()
project_id = "python-code-running"
table = f"projects/{project_id}/datasets/steam_reviews/tables/steam_reviews_dataset_medium"

# Use the BigQuery Storage API to read the data
requested_session = bigquery_storage.types.ReadSession()
requested_session.table = table
requested_session.data_format = bigquery_storage.types.DataFormat.ARROW

parent = f"projects/{project_id}"
session = client.create_read_session(
    parent=parent,
    read_session=requested_session,
    max_stream_count=1,
)

# Read the data
stream = session.streams[0]
reader = client.read_rows(stream.name)

# Convert to pandas DataFrame
arrow_table = reader.to_arrow()
df = arrow_table.to_pandas()
print("Data loaded successfully.")

# Preprocessing
print("Preprocessing data...")
df['voted_up'] = df['voted_up'].astype(int)
df_filtered = df[['author_steamid', 'appid', 'voted_up', 'game']]
df_filtered = df_filtered.drop_duplicates(subset=['author_steamid', 'appid'], keep='first')

# Get the top 10,000 games with the most reviews
top_games = df_filtered['appid'].value_counts().nlargest(25000).index
df_filtered = df_filtered[df_filtered['appid'].isin(top_games)]

# Display statistics before balancing
print("\nStatistics before balancing:")
print(f"Total number of reviews: {len(df_filtered)}")
print(f"Number of unique users: {df_filtered['author_steamid'].nunique()}")
print(f"Number of unique games: {df_filtered['appid'].nunique()}")
print("Value counts in 'voted_up':")
print(df_filtered['voted_up'].value_counts())

# Separate positive and negative samples
df_positive = df_filtered[df_filtered['voted_up'] == 1]
df_negative = df_filtered[df_filtered['voted_up'] == 0]

# Balance the dataset
n_samples = min(len(df_positive), len(df_negative))
df_positive_balanced = resample(df_positive, n_samples=n_samples, random_state=1)
df_negative_balanced = resample(df_negative, n_samples=n_samples, random_state=1)

# Combine balanced datasets
df_balanced = pd.concat([df_positive_balanced, df_negative_balanced])

# Display statistics after balancing
print("\nStatistics after balancing:")
print(f"Total number of reviews: {len(df_balanced)}")
print(f"Number of unique users: {df_balanced['author_steamid'].nunique()}")
print(f"Number of unique games: {df_balanced['appid'].nunique()}")
print("Value counts in 'voted_up':")
print(df_balanced['voted_up'].value_counts())

# Calculate average ratings per user
avg_ratings_per_user = df_balanced.groupby('author_steamid').size().mean()
print(f"\nAverage number of ratings per user: {avg_ratings_per_user:.2f}")

# Calculate sparsity
total_possible_ratings = df_balanced['author_steamid'].nunique() * df_balanced['appid'].nunique()
sparsity = 1 - (len(df_balanced) / total_possible_ratings)
print(f"Sparsity: {sparsity:.4f}")

# Create user and item mappings
user_ids = df_balanced['author_steamid'].unique()
item_ids = df_balanced['appid'].unique()
user_to_index = {uid: i for i, uid in enumerate(user_ids)}
item_to_index = {iid: i for i, iid in enumerate(item_ids)}


# Create PyTorch dataset
class SteamDataset(Dataset):
    def __init__(self, df, user_to_index, item_to_index):
        self.users = torch.tensor([user_to_index[u] for u in df['author_steamid']], dtype=torch.long)
        self.items = torch.tensor([item_to_index[i] for i in df['appid']], dtype=torch.long)
        self.ratings = torch.tensor(df['voted_up'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# Split the data
train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
print(f"\nNumber of samples in training set: {len(train_df)}")
print(f"Number of samples in validation set: {len(test_df)}")

train_dataset = SteamDataset(train_df, user_to_index, item_to_index)
test_dataset = SteamDataset(test_df, user_to_index, item_to_index)

# Create data loaders
batch_size = 2024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Define the SVD++ model
class SVDpp(nn.Module):
    def __init__(self, n_users, n_items, n_factors=75):
        super(SVDpp, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.user_implicit = nn.Embedding(n_users, n_factors)

    def forward(self, user, item):
        user_factor = self.user_factors(user)
        item_factor = self.item_factors(item)
        user_bias = self.user_biases(user).squeeze()
        item_bias = self.item_biases(item).squeeze()
        user_implicit_factor = self.user_implicit(user)

        prediction = (user_factor * item_factor).sum(1) + user_bias + item_bias
        return torch.sigmoid(prediction)

    def l2_regularization(self):
        return (self.user_factors.weight.norm(2) +
                self.item_factors.weight.norm(2) +
                self.user_biases.weight.norm(2) +
                self.item_biases.weight.norm(2) +
                self.user_implicit.weight.norm(2))


# Initialize the model
n_factors = 75
model = SVDpp(len(user_ids), len(item_ids), n_factors).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop with early stopping
n_epochs = 100
patience = 2
best_val_loss = float('inf')
counter = 0
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0
    for user, item, rating in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
        user, item, rating = user.to(device), item.to(device), rating.to(device)
        optimizer.zero_grad()
        prediction = model(user, item)
        loss = criterion(prediction, rating) + 1e-5 * model.l2_regularization()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for user, item, rating in test_loader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            prediction = model(user, item)
            loss = criterion(prediction, rating)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# Plot learning curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Load best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation
model.eval()
predictions = []
true_ratings = []
with torch.no_grad():
    for user, item, rating in tqdm(test_loader, desc="Evaluating"):
        user, item = user.to(device), item.to(device)
        prediction = model(user, item)
        predictions.extend(prediction.cpu().numpy())
        true_ratings.extend(rating.numpy())

predictions = np.array(predictions)
true_ratings = np.array(true_ratings)


# Calculate metrics
def calculate_metrics(predictions, true_ratings, top_n=10):
    user_est_true = defaultdict(list)
    for i, (uid, iid) in enumerate(zip(test_df['author_steamid'], test_df['appid'])):
        user_est_true[uid].append((iid, predictions[i], true_ratings[i]))

    MAP_at_K = []
    NDCG_at_K = []
    HR_at_K = []

    for uid, user_ratings in tqdm(user_est_true.items(), desc="Calculating metrics"):
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n_items = user_ratings[:top_n]

        pred = [iid for (iid, _, _) in top_n_items]
        actual = [iid for (iid, _, true_r) in user_ratings if true_r >= 0.5]

        if len(actual) == 0:
            continue

        # MAP@K
        ap_at_k = sum([1 if iid in actual else 0 for iid in pred]) / min(len(actual), top_n)
        MAP_at_K.append(ap_at_k)

        # NDCG@K
        dcg = sum([1 / np.log2(i + 2) if iid in actual else 0 for i, (iid, _, _) in enumerate(top_n_items)])
        ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), top_n))])
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        NDCG_at_K.append(ndcg)

        # HR@K
        hr_at_k = 1 if any(iid in actual for (iid, _, _) in top_n_items) else 0
        HR_at_K.append(hr_at_k)

    return {
        f'MAP@{top_n}': np.mean(MAP_at_K),
        f'NDCG@{top_n}': np.mean(NDCG_at_K),
        f'HR@{top_n}': np.mean(HR_at_K)
    }


metrics = calculate_metrics(predictions, true_ratings)
print("\nMetrics for SVD++ model:")
for metric, value in metrics.items():
    print(f'{metric}: {value:.4f}')


# Function to generate recommendations
def generate_recommendations(model, user_id, game_ids, game_names, user_history, n=10):
    model.eval()
    with torch.no_grad():
        user = torch.tensor([user_to_index[user_id]]).to(device)
        items = torch.tensor([item_to_index[iid] for iid in game_ids if iid in item_to_index]).to(device)
        predictions = model(user.repeat(len(items)), items)

    # Create a dictionary of game_id to prediction
    game_predictions = {game_id: pred.item() for game_id, pred in zip(game_ids, predictions) if
                        game_id not in user_history}

    # Sort by prediction score and get top N
    top_n = sorted(game_predictions.items(), key=lambda x: x[1], reverse=True)[:n]

    return [(game_id, game_names[game_id]) for game_id, _ in top_n]


# Example: Generate recommendations for a specific user
example_user = df_balanced['author_steamid'].iloc[0]
all_game_ids = df_balanced['appid'].unique()
game_names = df[['appid', 'game']].drop_duplicates().set_index('appid')['game'].to_dict()
user_history = set(df_balanced[df_balanced['author_steamid'] == example_user]['appid'])

recommendations = generate_recommendations(model, example_user, all_game_ids, game_names, user_history)

print(f"\nTop 10 Recommended Games for User {example_user}:")
for game_id, game_name in recommendations:
    print(f"Game: {game_name}")

print("\nTraining completed.")
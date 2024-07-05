import pandas as pd
import torch
from torch import nn
import numpy as np
from google.cloud import bigquery_storage


# Define the SVD++ model
class SVDpp(nn.Module):
    def __init__(self, n_users, n_items, n_factors=100):
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

        prediction = (user_factor * item_factor).sum(1) + user_bias + item_bias + (
                    user_implicit_factor * item_factor).sum(1)
        return torch.sigmoid(prediction)


def load_model(model_path, device='cpu'):
    state_dict = torch.load(model_path, map_location=device)
    n_users = state_dict['user_factors.weight'].size(0)
    n_items = state_dict['item_factors.weight'].size(0)

    model = SVDpp(n_users, n_items)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def load_game_list_from_bigquery():
    print("Loading game list from BigQuery...")
    client = bigquery_storage.BigQueryReadClient()
    project_id = "python-code-running"
    table = f"projects/{project_id}/datasets/steam_reviews/tables/steam_reviews_dataset_small"

    requested_session = bigquery_storage.types.ReadSession()
    requested_session.table = table
    requested_session.data_format = bigquery_storage.types.DataFormat.ARROW

    parent = f"projects/{project_id}"
    session = client.create_read_session(
        parent=parent,
        read_session=requested_session,
        max_stream_count=1,
    )

    stream = session.streams[0]
    reader = client.read_rows(stream.name)

    arrow_table = reader.to_arrow()
    df = arrow_table.to_pandas()

    # Preprocessing
    print("Preprocessing data...")
    df['voted_up'] = df['voted_up'].astype(int)
    df = df[['author_steamid', 'appid', 'voted_up', 'game']]
    df = df.drop_duplicates(subset=['author_steamid', 'appid'], keep='first')

    # Get the top N games with the most reviews
    top_games = df['appid'].value_counts().nlargest(15000).index
    df = df[df['appid'].isin(top_games)]

    print("Game list loaded successfully.")

    # Count the number of reviews for each game
    game_popularity = df['game'].value_counts()

    return game_popularity


def load_user_ratings(file_path):
    print(f"Loading user ratings from {file_path}...")
    df = pd.read_csv(file_path)
    print("User ratings loaded successfully.")
    return df


def generate_recommendations(model, user_ratings, game_popularity, game_to_index, device, n=15):
    model.eval()
    user_id = 0  # We'll use 0 as the user_id for the new user
    user_history = set(user_ratings['game'])
    n_items = model.item_factors.num_embeddings

    # Create a reverse mapping
    index_to_game = {v: k for k, v in game_to_index.items()}

    with torch.no_grad():
        user = torch.tensor([user_id]).to(device)
        valid_items = [game_to_index[game] for game in game_popularity.index
                       if game not in user_history
                       and game in game_to_index
                       and game_to_index[game] < n_items]

        if not valid_items:
            print("No valid games found for recommendations.")
            return []

        items = torch.tensor(valid_items).to(device)
        predictions = model(user.repeat(len(items)), items)

    game_predictions = {index_to_game[idx.item()]: pred.item()
                        for idx, pred in zip(items, predictions)}

    # Sort by prediction score, then by popularity
    top_n = sorted(game_predictions.items(), key=lambda x: (x[1], game_popularity.get(x[0], 0)), reverse=True)[:n]

    return top_n


def main():
    device = torch.device("cpu")

    # Load the game list and popularity from BigQuery
    game_popularity = load_game_list_from_bigquery()

    # Load the model
    model = load_model('final_steam_model.pth', device)

    # Create game_to_index mapping
    game_to_index = {game: idx for idx, game in enumerate(game_popularity.index)}

    # Load user ratings from CSV
    user_ratings = load_user_ratings('user_game_preferences.csv')

    # Print user rating statistics
    total_ratings = len(user_ratings)
    positive_ratings = user_ratings['voted_up'].sum()
    negative_ratings = total_ratings - positive_ratings

    print(f"Number of user ratings: {total_ratings}")
    print(f"Positive ratings: {positive_ratings}")
    print(f"Negative ratings: {negative_ratings}")

    # Generate recommendations
    recommendations = generate_recommendations(model, user_ratings, game_popularity, game_to_index, device)

    if recommendations:
        print(f"\nTop 15 Recommended Games for the User:")
        for game, score in recommendations:
            print(f"Game: {game}, Prediction Score: {score:.4f}, Popularity: {game_popularity.get(game, 0)} reviews")
    else:
        print("No recommendations could be generated.")


if __name__ == "__main__":
    main()
# Steam Game Recommendation System

## Overview
This project showcases a collaborative filtering-based recommendation system designed to suggest Steam games to users based on their review preferences. The methodology incorporates SVD++ model, and operates on a dataset of Steam reviews, focusing on user interactions with various games.

## Data
The dataset includes reviews from multiple users across numerous Steam games, classified into positive and negative reviews. Here are some statistics about the dataset before and after preprocessing:

### Preprocessing Statistics

#### Statistics Before Balancing
| Description                | N Samples   |
|----------------------------|-------------|
| Total number of reviews    | 19,173,029  |
| Number of unique users     | 1,795,686   |
| Number of unique games     | 25,000      |
| Positive reviews (voted_up)| 16,036,147  |
| Negative reviews           | 3,136,882   |

#### Statistics After Balancing
| Description                | N Samples   |
|----------------------------|-------------|
| Total number of reviews    | 6,273,764   |
| Number of unique users     | 1,522,902   |
| Number of unique games     | 24,999      |
| Positive reviews (voted_up)| 3,136,882   |
| Negative reviews           | 3,136,882   |

### Sparsity: 0.9998

## Recommender Model
### Collaborative Filtering (CF)
Collaborative Filtering is a technique used in recommendation systems where the system predicts a user’s preferences based on the preferences of other users with similar tastes. Our implementation leverages user-item interaction data, particularly reviews indicating whether a user liked (voted up) or disliked (voted down) a game.

### Describing the SVD++ Model
SVD++ is an enhancement over the traditional singular value decomposition (SVD) method for collaborative filtering. It extends SVD by taking into account implicit feedback (e.g., all the items reviewed by a user, regardless of rating). It factors in both explicit interactions (ratings) and implicit interactions (review history), making the model better at handling sparse datasets and providing more personalized recommendations.

## Implementation
The model's implementation involved the following steps:
1. Preprocessing the input data to balance the number of positive and negative reviews for a more unbiased training process.
2. Splitting data into training and validation sets.
3. Training the SVD++ model using the training dataset with an early stopping mechanism to prevent overfitting.
4. Evaluating the model on the validation set.

## Evaluation Metrics
| Metric | Value    |
|--------|----------|
| MAP@10 | 0.9991   |
| NDCG@10| 0.9725   |
| HR@10  | 0.9999   |

## Recommendations
Example recommendations for User 76561199360139694:

| Game                                |
|-------------------------------------|
| Portal 2                            |
| Wallpaper Engine                    |
| Vampire Survivors                   |
| RimWorld                            |
| ULTRAKILL                           |
| Hades                               |
| South Park™: The Stick of Truth™    |
| Half-Life 2                         |
| Left 4 Dead 2                       |
| Resident Evil 2                     |

These recommendations are generated based on the user's previous interactions with various games available on Steam, aiming to enrich their gaming experience by suggesting titles likely to be of interest.

## Conclusion
The developed recommendation system demonstrates the potential to provide highly accurate game suggestions on the Steam platform using collaborative filtering with the SVD++ model. It efficiently handles a large dataset by focusing on balancing and optimizing interactions, making the system robust and reliable for real-world applications.

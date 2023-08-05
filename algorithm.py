# Text classification with supervised learning

import re
import praw
import config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

## Loading data

# List of themes to search on reddit
# These are the classes we will be using as target variables
matters = ['datascience', 'machinelearning']

# Function to load data
def load_data():

    # Connecting to the reddit API
    reddit_api = praw.Reddit(
        client_id='your client id',
        client_secret='your client secret',
        password='your password',
        user_agent='your user agent',
        username='your username'
    )

    # Counting the number of characters using regular expressions
    char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))

    # Defining the condition to filter the posts
    mask = lambda post: char_count(post) >= 100

    # Lists for results
    data = []
    labels = []

    # Loop
    for i, matter in enumerate(matters):

        # Extracting the posts
        subreddit_data = reddit_api.subreddit(matter).new(limit=1000)

        # Filtering the posts that do not satisfy our condition
        posts = [post.selftext for post in filter(mask, subreddit_data)]

        # Adding posts and labels to lists
        data.extend(posts)
        labels.extend([i] * len(posts))

        # Showing
        print(
            f'Number os posts of the matter r/{matter}: {len(posts)}',
            f'\nOne of the extracted posts: {posts[0][:600]}...\n',
            '_' * 80 + '\n'
        )
    return data, labels

## Division into training and test data

# Control variables
TEST_SIZE = .2
RANDOM_STATE = 0

# Function to split data
def split_data():

    print(f'Split {100 * TEST_SIZE}% of data for model testing and evaluation...')

    # Split of data
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f'{len(y_test)} test samples')
    
    return x_train, x_test, y_train, y_test

## Data pre-processing and attribute extraction

# > Remove symbols, numbers and url-like strings with custom preprocessor
# > Vectorize text using the term inverse frequency of the document frequency (TfidfVectorizer)
# > Reduces to prime values using singular value decomposition
# > Partition data and labels into training/validation sets

# Control variables
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30

# Function for the preprocessing pipeline
def preprocessing_pipeline():

    # Remove non-alphabetic character
    pattern = r'\W|\d|http.*\s+|www.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    # Vectorization TF-IDF
    vectorizer = TfidfVectorizer(preprocessor=preprocessor, stop_words='english', min_df=MIN_DOC_FREQ)
    # Reducing the dimensionality of the TF-IDF vector
    decomposition = TruncatedSVD(n_components=N_COMPONENTS, n_iter=N_ITER)

    # Pipeline
    pipeline = [('tf-idf', vectorizer), ('svd', decomposition)]

    return pipeline

## Models selection

# Control variables
N_NEIGHBORS = 4
CV = 3

# Function to build the models
def build_models():

    model_1 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    model_2 = RandomForestClassifier(random_state=RANDOM_STATE)
    model_3 = LogisticRegressionCV(cv=CV, random_state=RANDOM_STATE)

    models = [('KNN', model_1), ('RandomForest', model_2), ('LogReg', model_3)]
    return models

## Training and evaluation of models

# Function for training and evaluation of models
def train_evaluate(models, pipeline, x_test, x_train, y_test, y_train):

    results = []

    # Loop
    for name, model in models:

        # Pipeline
        pipe = Pipeline(pipeline + [(name, model)])

        # Training
        print(f'Training the model {model} with the data train')
        pipe.fit(x_train, y_train)

        # Predicts
        y_pred = pipe.predict(x_test)

        # Calculates the metrics
        report = classification_report(y_test, y_pred)
        print(f'Relatory of classifications\n', report)

        results.append([model, {'model':name, 'predict':y_pred, 'report':report,}])

    return results

## Running the pipeline for all models

# Machine Learning Pipeline
if __name__ == '__main__':

    # Loading data
    data, labels = load_data()

    # Split
    x_train, x_test, y_train, y_test = split_data()

    # Preprocessing pipeline
    pipeline = preprocessing_pipeline()

    # Build the models
    all_models = build_models()

    # Train and Evaluate the models
    results = train_evaluate(all_models, pipeline, x_train, x_test, y_train, y_test)

print('Successfully Concluded')
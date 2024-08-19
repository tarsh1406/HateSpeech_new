import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import json

def load_data():
    dataset1 = pd.read_csv('data/dataset1.csv')
    dataset2 = pd.read_csv('data/dataset2.csv')
    combined_dataset = pd.concat([dataset1, dataset2])
    combined_dataset.dropna(subset=['Text', 'Type'], inplace=True)
    combined_dataset['Type'] = combined_dataset['Type'].apply(lambda x: 0 if 'appropriate' in x else 1)
    return combined_dataset['Text'], combined_dataset['Type']

def train_and_save_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = {
        'svm': SVC(),
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100)
    }

    training_results = {}
    testing_results = {}
    
    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        
        # Training results
        y_train_pred = model.predict(X_train_tfidf)
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        training_results[model_name] = train_report
        
        # Testing results
        y_test_pred = model.predict(X_test_tfidf)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        testing_results[model_name] = test_report
        
        print(f"Model: {model_name}")
        print("Training results:")
        print(classification_report(y_train, y_train_pred))
        print("Testing results:")
        print(classification_report(y_test, y_test_pred))

        # Save each model
        model_path = f'models/{model_name}_model.pkl'
        joblib.dump(model, model_path)

    # Save the vectorizer
    vectorizer_path = 'models/vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_path)

    # Save classification reports
    with open('models/training_results.json', 'w') as file:
        json.dump(training_results, file)

    with open('models/testing_results.json', 'w') as file:
        json.dump(testing_results, file)

    return training_results, testing_results

if __name__ == '__main__':
    training_results, testing_results = train_and_save_models()
    for model_name in training_results:
        print(f'{model_name} training accuracy: {training_results[model_name]["accuracy"]:.4f}')
        print(f'{model_name} testing accuracy: {testing_results[model_name]["accuracy"]:.4f}')

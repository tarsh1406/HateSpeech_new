import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    dataset1 = pd.read_csv('data/dataset1.csv')
    dataset2 = pd.read_csv('data/dataset2.csv')
    combined_dataset = pd.concat([dataset1, dataset2])
    combined_dataset.dropna(subset=['Text', 'Type'], inplace=True)
    combined_dataset['Type'] = combined_dataset['Type'].apply(lambda x: 0 if 'appropriate' in x else 1)
    return combined_dataset['Text'], combined_dataset['Type']

def train_and_evaluate():
    X, y = load_data()
    training_sizes = [0.8, 0.7, 0.6, 0.5, 0.4]
    results = []
    
    for size in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'Training Size': size,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/logistic_regression_accuracy_results.csv', index=False)
    
    # Plot the graph
    plt.figure()
    plt.plot(training_sizes, results_df['Accuracy'], label='Accuracy', marker='o')
    plt.plot(training_sizes, results_df['Precision'], label='Precision', marker='o')
    plt.plot(training_sizes, results_df['Recall'], label='Recall', marker='o')
    plt.plot(training_sizes, results_df['F1 Score'], label='F1 Score', marker='o')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Logistic Regression Evaluation Metrics vs Training Size')
    plt.legend()
    plt.savefig('models/logistic_regression_evaluation_metrics_graph.png')  # Save the graph
    plt.show()

if __name__ == '__main__':
    train_and_evaluate()

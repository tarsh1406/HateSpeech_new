import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
    svm_accuracies = []
    nb_accuracies = []
    lr_accuracies = []
    rf_accuracies = []
    
    results = []
    
    for size in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        svm = SVC()
        nb = MultinomialNB()
        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=100)
        
        svm.fit(X_train_tfidf, y_train)
        nb.fit(X_train_tfidf, y_train)
        lr.fit(X_train_tfidf, y_train)
        rf.fit(X_train_tfidf, y_train)
        
        svm_acc = accuracy_score(y_test, svm.predict(X_test_tfidf))
        nb_acc = accuracy_score(y_test, nb.predict(X_test_tfidf))
        lr_acc = accuracy_score(y_test, lr.predict(X_test_tfidf))
        rf_acc = accuracy_score(y_test, rf.predict(X_test_tfidf))
        
        svm_accuracies.append(svm_acc)
        nb_accuracies.append(nb_acc)
        lr_accuracies.append(lr_acc)
        rf_accuracies.append(rf_acc)
        
        results.append({
            'Training Size': size,
            'SVM Accuracy': svm_acc,
            'Naive Bayes Accuracy': nb_acc,
            'Logistic Regression Accuracy': lr_acc,
            'Random Forest Accuracy': rf_acc
        })
    
    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/accuracy_results.csv', index=False)
    
    # Plot the graph
    plt.plot(training_sizes, svm_accuracies, label='SVM', marker='o')
    plt.plot(training_sizes, nb_accuracies, label='Naive Bayes', marker='o')
    plt.plot(training_sizes, lr_accuracies, label='Logistic Regression', marker='o')
    plt.plot(training_sizes, rf_accuracies, label='Random Forest', marker='o')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('models/accuracy_graph.png')  # Save the graph
    plt.show()

if __name__ == '__main__':
    train_and_evaluate()

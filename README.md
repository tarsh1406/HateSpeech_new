HATE SPEECH CLASSIFICATION FOR ONLINE VIDEO FEEDBACK USING MACHINE LEARNING APPROACH

Project Overview  
This project focuses on detecting and classifying hate speech in online video feedback using machine learning techniques. The feedback data is primarily sourced from YouTube video comments, which are analyzed and categorized as either hate speech or non-hate speech. The system also provides finer classifications into specific categories such as racism, sexism, or religious discrimination. The overall goal is to contribute to a safer and more respectful online community by identifying harmful language in user-generated content.

Key Features
Automated Hate Speech Detection: Analyzes YouTube video comments and flags potentially harmful content using machine learning models.
Multi-class Classification: Classifies hate speech into various categories, such as racism, sexism, and other forms of discrimination.
User-Friendly Web Dashboard: Provides an intuitive interface for uploading and analyzing video comments, along with visual insights.
Scalable Framework: The solution is designed to be scalable and adaptable for future enhancements and integration with additional data sources.

Dataset
The dataset consists of labeled video comments collected primarily from YouTube. The data is preprocessed to remove noise and enhance the quality of the model inputs. Labels include categories such as hate speech, non-hate speech, and various subcategories for more specific classification.

Model and Methodology
This project applies Natural Language Processing (NLP) techniques, including tokenization, stemming, and TF-IDF, for feature extraction. The classification model is built using algorithms like Support Vector Machines (SVM), Naive Bayes, Logistic Regression and Random Forest. The model is evaluated based on accuracy, precision, recall, and F1-score.

Results and Performance
The model has achieved an accuracy on the test set, with detailed evaluation metrics available in the results/ directory. Visualization of the classification performance and confusion matrices are also provided.

Future Enhancements
Real-time Integration: Implementing real-time classification for live video comment streams.
Expanded Dataset: Including more diverse and representative data to improve model generalization.
Advanced Model Architectures: Experimenting with transformer models like BERT for enhanced performance.


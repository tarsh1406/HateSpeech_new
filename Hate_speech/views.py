from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views
from django.contrib.auth.models import User
from django.contrib.auth import update_session_auth_hash
from django.contrib import messages
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.hashers import make_password
from django.core.mail import send_mail
from .models import History, Comment, CustomUser
from .forms import CustomUserCreationForm
import random
import string
import requests
from django.conf import settings
from django.contrib.auth import logout
import pickle
from django.http import JsonResponse
import os
import json
import joblib
from .train_model import train_and_save_models
from urllib.parse import urlparse, parse_qs
import urllib.parse as urlparse
import re
from googleapiclient.discovery import build
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from django.utils.timezone import localtime

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


def get_classification_report(y_true, y_pred, title):
    report = classification_report(y_true, y_pred, output_dict=True)
    return report



# Set the path to the vectorizer.pkl and model files
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorizer_path = os.path.join(current_dir, 'models', 'vectorizer.pkl')
svm_model_path = os.path.join(current_dir, 'models', 'svm_model.pkl')
nb_model_path = os.path.join(current_dir, 'models', 'naive_bayes_model.pkl')
lr_model_path = os.path.join(current_dir, 'models', 'logistic_regression_model.pkl')
rf_model_path = os.path.join(current_dir, 'models', 'random_forest_model.pkl')
training_results_path = os.path.join(current_dir, 'models', 'training_results.json')
testing_results_path = os.path.join(current_dir, 'models', 'testing_results.json')

# Load the vectorizer and models using joblib
vectorizer = joblib.load(vectorizer_path)
svm_model = joblib.load(svm_model_path)
nb_model = joblib.load(nb_model_path)
lr_model = joblib.load(lr_model_path)
rf_model = joblib.load(rf_model_path)


def index_view(request):
    return render(request, 'index.html')

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                form.add_error(None, "Invalid username or password.")  # Add a non-field error
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
        # If form is not valid, reload the page with the form and errors
        return render(request, 'register.html', {'form': form})
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})

class PasswordResetView(auth_views.PasswordResetView):
    template_name = 'password_reset.html'

class PasswordResetDoneView(auth_views.PasswordResetDoneView):
    template_name = 'password_reset_done.html'

class PasswordResetConfirmView(auth_views.PasswordResetConfirmView):
    template_name = 'password_reset_confirm.html'

class PasswordResetCompleteView(auth_views.PasswordResetCompleteView):
    template_name = 'password_reset_complete.html'

@login_required
def edit_profile(request):
    if request.method == 'POST':
        new_username = request.POST['username']
        new_email = request.POST['email']
        user = request.user
        user.username = new_username
        user.email = new_email
        user.save()

        if 'password' in request.POST and request.POST['password']:
            password_form = PasswordChangeForm(user=user, data={
                'old_password': request.POST['password'],
                'new_password1': request.POST['password'],
                'new_password2': request.POST['password']
            })
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)  # Important!
                messages.success(request, 'Password updated successfully.')
            else:
                messages.error(request, password_form.errors)
        else:
            messages.success(request, 'Profile updated successfully')
        return redirect('home')

    return render(request, 'edit_profile.html', {'user': request.user})

@login_required
def history(request):
    if request.method == 'POST' and 'delete_id' in request.POST:
        delete_id = request.POST['delete_id']
        History.objects.filter(id=delete_id).delete()
        return redirect('history')

    history_entries = History.objects.all().order_by('-created_at')
    return render(request, 'history.html', {'history': history_entries})

@login_required
def dashboard(request):
    video_details = request.session.get('video_details')
    comments = []
    accuracies = {}
    training_results = {}
    testing_results = {}
    hate_speech_percentage = 0

    if 'youtubeLink' in request.GET:
        video_id = extract_video_id(request.GET['youtubeLink'])
        if not video_id:
            messages.error(request, "Invalid YouTube link. Please provide a valid link.")
            return redirect('dashboard')

        api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={settings.YOUTUBE_API_KEY}&part=snippet"
        response = requests.get(api_url)
        if response.status_code == 200:
            try:
                response_json = response.json()
                if 'items' in response_json and len(response_json['items']) > 0:
                    video_details = response_json['items'][0]['snippet']
                    request.session['video_details'] = video_details

                    title = video_details['title']
                    history_entry = History(video_link=request.GET['youtubeLink'], video_title=title)
                    history_entry.save()

                    comments_response = get_video_comments(video_id)
                    analyzed_comments, hate_speech_count = analyze_comments(comments_response)
                    accuracies = get_accuracies_and_graph()

                    # Calculate hate speech percentage
                    if analyzed_comments:
                        hate_speech_percentage = (hate_speech_count / len(analyzed_comments)) * 100

                    # Load training results
                    with open(training_results_path, 'r') as file:
                        training_results = json.load(file)

                    # Load testing results
                    with open(testing_results_path, 'r') as file:
                        testing_results = json.load(file)

                    return render(request, 'dashboard.html', {
                        'video_details': video_details,
                        'comments': analyzed_comments,
                        'accuracies': accuracies,
                        'training_results': training_results,
                        'testing_results': testing_results,
                        'hate_speech_percentage': hate_speech_percentage
                    })
                else:
                    messages.error(request, "Failed to fetch video details or no details found.")
            except json.JSONDecodeError:
                messages.error(request, "Failed to decode the response from YouTube.")
        else:
            messages.error(request, "Failed to fetch video details from YouTube.")

    # Get updated history if no new video link is provided
    history = History.objects.all().order_by('-created_at')

    return render(request, 'dashboard.html', {
        'video_details': video_details,
        'comments': comments,
        'accuracies': accuracies,
        'training_results': training_results,
        'testing_results': testing_results,
        'hate_speech_percentage': hate_speech_percentage
    })




# Define the function to extract video ID from the URL
def extract_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    Supports both standard and short URLs.
    """
    # Regular expression for YouTube video ID
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def predict_video(request):
    if request.method == 'POST':
        url = request.POST.get('url')
        video_id = extract_video_id(url)
        if not video_id:
            return JsonResponse({'error': 'Invalid URL'}, status=400)
        
        # Assuming you have a function to process the video ID and get prediction
        result = process_video_and_predict(video_id)  # Replace this with actual processing function
        
        return JsonResponse({'result': result})
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def analyze_comments(comments):
    results = []
    hate_speech_count = 0
    for comment in comments:
        print(comment)  # Print the comment structure for debugging
        try:
            # Ensure that the comment structure is correct before accessing it
            snippet = comment.get('snippet', {})
            topLevelComment = snippet.get('topLevelComment', {})
            comment_snippet = topLevelComment.get('snippet', {})
            comment_text = comment_snippet.get('textDisplay', '')

            # Check if comment_text is empty, and if so, skip this comment
            if not comment_text:
                continue

            transformed_comment = vectorizer.transform([comment_text])
            svm_pred = svm_model.predict(transformed_comment)
            nb_pred = nb_model.predict(transformed_comment)
            lr_pred = lr_model.predict(transformed_comment)
            rf_pred = rf_model.predict(transformed_comment)

            is_hate_speech = any([svm_pred[0], nb_pred[0], lr_pred[0], rf_pred[0]])
            if is_hate_speech:
                hate_speech_count += 1

            results.append({
                'comment': comment_text,
                'svm': svm_pred[0],
                'naive_bayes': nb_pred[0],
                'logistic_regression': lr_pred[0],
                'random_forest': rf_pred[0],
                'is_hate_speech': 'Yes' if is_hate_speech else 'No'
            })
        except KeyError as e:
            print(f"KeyError: {e} in comment: {comment}")
        except TypeError as e:
            print(f"TypeError: {e} in comment: {comment}")
    return results, hate_speech_count





def get_accuracies_and_graph():
    accuracies = {
        'svm': 0.8614,
        'naive_bayes': 0.8059,
        'logistic_regression': 0.8298,
        'random_forest': 0.8569
    }
    return accuracies

def forgot_password(request):
    if request.method == 'POST' and 'email' in request.POST:
        email = request.POST['email']
        try:
            user = CustomUser.objects.get(email=email)
            token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=50))
            user.reset_token = token
            user.save()

            reset_link = request.build_absolute_uri(f'/reset_password/?token={token}')
            send_mail(
                'Password Reset',
                f'Password reset link: {reset_link}',
                'from@example.com',
                [email],
                fail_silently=False,
            )
            messages.success(request, "Password reset link has been sent to your email.")
        except CustomUser.DoesNotExist:
            messages.error(request, "No user found with that email address.")

    return render(request, 'forgot_password.html')

def reset_password(request):
    token = request.GET.get('token')
    if request.method == 'POST' and 'password' in request.POST:
        new_password = make_password(request.POST['password'])
        try:
            user = CustomUser.objects.get(reset_token=token)
            user.password = new_password
            user.reset_token = ''
            user.save()
            messages.success(request, "Password reset successfully. You can now log in with your new password.")
            return redirect('login')
        except CustomUser.DoesNotExist:
            messages.error(request, "Invalid reset token.")

    return render(request, 'reset_password.html', {'token': token})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def train_view(request):
    # Load your training and testing data
    X, y = load_data()  # Ensure load_data() is correctly implemented to load your dataset
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

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_train_pred = model.predict(X_train_tfidf)
        y_test_pred = model.predict(X_test_tfidf)

        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        accuracies[name] = {
            'train_cm': plot_confusion_matrix(cm_train, f'{name} Train Confusion Matrix'),
            'test_cm': plot_confusion_matrix(cm_test, f'{name} Test Confusion Matrix'),
            'train_report': get_classification_report(y_train, y_train_pred, f'{name} Train Classification Report'),
            'test_report': get_classification_report(y_test, y_test_pred, f'{name} Test Classification Report'),
        }

    # Save the vectorizer
    vectorizer_path = os.path.join(current_dir, 'models', 'vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)

    # Save classification reports
    with open(training_results_path, 'w') as file:
        json.dump({name: accuracies[name]['train_report'] for name in models}, file)

    with open(testing_results_path, 'w') as file:
        json.dump({name: accuracies[name]['test_report'] for name in models}, file)

    return JsonResponse({'status': 'Training completed.', 'accuracies': accuracies})


@login_required
def chart_data(request):
    with open(training_results_path, 'r') as file:
        training_results = json.load(file)
    with open(testing_results_path, 'r') as file:
        testing_results = json.load(file)

    accuracies = {
        'svm': {
            'training': training_results['svm']['accuracy'],
            'testing': testing_results['svm']['accuracy']
        },
        'naive_bayes': {
            'training': training_results['naive_bayes']['accuracy'],
            'testing': testing_results['naive_bayes']['accuracy']
        },
        'logistic_regression': {
            'training': training_results['logistic_regression']['accuracy'],
            'testing': testing_results['logistic_regression']['accuracy']
        },
        'random_forest': {
            'training': training_results['random_forest']['accuracy'],
            'testing': testing_results['random_forest']['accuracy']
        }
    }
    return JsonResponse(accuracies)


def normalize_youtube_url(url):
    # Extract video ID from both types of URLs
    match = re.search(r'(?:v=|live/|embed/|youtu.be/|watch\?v=)([^&?/]+)', url)
    if match:
        video_id = match.group(1)
        return f'https://www.youtube.com/watch?v={video_id}'
    else:
        raise ValueError('Invalid YouTube URL')


def get_video_comments(video_id):
    api_key = settings.YOUTUBE_API_KEY
    youtube = build('youtube', 'v3', developerKey=api_key)

    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100
    ).execute()

    while response:
        comments.extend(response['items'])
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=response['nextPageToken'],
                maxResults=100
            ).execute()
        else:
            break

    return comments




def analyze_comment(comment, model, vectorizer):
    # Transform comment using vectorizer
    comment_vector = vectorizer.transform([comment])
    # Predict using the model
    prediction = model.predict(comment_vector)
    return prediction[0]  # Assuming 0: No Hate Speech, 1: Hate Speech

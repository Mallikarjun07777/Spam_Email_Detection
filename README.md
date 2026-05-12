# Spam Email Detection System

A simple machine learning project that detects whether a message is **Spam** or **Not Spam**. The project uses a trained Naive Bayes classifier with text vectorization and provides an easy-to-use Streamlit web interface.

## Project Overview

This project classifies text messages or email-style content into two categories:

- **Spam**: Unwanted or promotional messages
- **Not Spam**: Normal useful messages

The model is trained using the SMS Spam Collection dataset and saved as pickle files. The Streamlit app loads the saved model and vectorizer to make predictions from user input.

## Features

- Detects spam messages using machine learning
- Streamlit-based user interface
- Uses `CountVectorizer` for text feature extraction
- Uses `MultinomialNB` Naive Bayes classifier
- Includes a separate training script
- Saved model and vectorizer are included in the `model` folder

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Pickle

## Project Structure

```text
Spam_Email_Detection/
+-- app.py
+-- main.py
+-- requirements.txt
+-- README.md
+-- model/
    +-- spam_model.pkl
    +-- vectorizer.pkl
```

## File Description

| File/Folder | Description |
| --- | --- |
| `app.py` | Streamlit web application for spam prediction |
| `main.py` | Script for loading dataset, training model, and testing predictions |
| `model/spam_model.pkl` | Saved trained spam detection model |
| `model/vectorizer.pkl` | Saved text vectorizer used for transforming input text |
| `requirements.txt` | Python libraries required to run the project |

## Dataset

The training script uses the SMS Spam Collection dataset from:

```text
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
```

The dataset contains labeled messages:

- `ham`: Not spam
- `spam`: Spam

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd Spam_Email_Detection
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run the Application

Run the Streamlit app using:

```bash
streamlit run app.py
```

After running the command, the app will open in your browser. Enter a message in the text box and click **Predict** to check whether it is spam or not.

## How the Model Works

1. The dataset is loaded from an online CSV/TSV source.
2. Labels are converted into numeric values:
   - `ham` becomes `0`
   - `spam` becomes `1`
3. The dataset is split into training and testing data.
4. Text messages are converted into numerical features using `CountVectorizer`.
5. A `MultinomialNB` model is trained.
6. The model is evaluated using accuracy score and classification report.
7. The saved model and vectorizer are used by the Streamlit app for prediction.

## Example Predictions

| Message | Prediction |
| --- | --- |
| `Congratulations! You won a free lottery ticket` | Spam |
| `Hey, are we meeting today?` | Not Spam |

## GitHub Upload Steps

After creating a new GitHub repository, use these commands:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

If the repository is already initialized, use:

```bash
git remote add origin <your-repository-url>
git add .
git commit -m "Add spam email detection project"
git push -u origin main
```

## Future Improvements

- Add more preprocessing steps such as stopword removal and stemming
- Improve model accuracy using TF-IDF vectorization
- Add email subject/body input fields
- Deploy the app online using Streamlit Community Cloud

## Author

Mallikarjun Hiremath

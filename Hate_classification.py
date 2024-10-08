#general packages for data manipulation
import pandas as pd
import numpy as np
import re
from collections import Counter#visualizations
import matplotlib.pyplot as plt
import seaborn as sns
#text preprocessing libraries
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
# sklearn libraries for feature extraction and model building
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score
#handle the warnings in the code
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning)

# Data Preprocessing
#import data
tweet_data = pd.read_csv('TwitterHate.csv')
print('The first 5 rows of the dataset are:\n',tweet_data.head())

# Data Cleaning
# Drop the id column and NA values
tweet_data.drop('id',axis=1,inplace=True)
tweet_data = tweet_data.dropna()

# Normalise the casing of the tweet
tweet_data['tweet'] = tweet_data['tweet'].str.lower()

# Normalise terms with dilect characters. NFKD is a normalisation form that splits the base character from its diacritic and then converts it into ascii.
tweet_data['tweet'] = tweet_data['tweet'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# Removing user handles starting with @
tweet_data['tweet'] = tweet_data['tweet'].replace(r'@\w+', '', regex=True)

# Removing URLs
tweet_data['tweet']= tweet_data['tweet'].replace(r'http\S+', '', regex=True)

# Using TweetTokenizer from NLTK, tokenize the tweets into individual terms
tokenizer = TweetTokenizer(preserve_case=True)
tweet_data['tweet'] = tweet_data['tweet'].apply(tokenizer.tokenize)

# Remove english stopwords and additional words like amp, rt, etc. from the tweets using NLTK stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['amp', 'rt', 'via', 'retweet','ur', 'u', 'w', 'b', 'c', 'im', 'dont', 'cant', 'thats', 'wont', 'isnt', 'didnt', 'couldnt', 'wouldnt'])
tweet_data['tweet'] = tweet_data['tweet'].apply(lambda x: [item for item in x if not item in stop_words])

# Remove punctuation, special characters and numbers from the tweet terms
tweet_data['tweet'] = tweet_data['tweet'].apply(lambda x: [item for item in x if item.isalpha()])


# Removeing ‘#’ symbols from the tweet terms and retaining the term
def remove_hashsymbols(text):
    pattern = re.compile(r'#') # re.compile() is used to convert a regular expression pattern string into a special re object
    text = ' '.join(text) # joining the list of tokens into a string
    clean_text = re.sub(pattern,'',text) # using re.sub() to replace the pattern with an empty string
    return tokenizer.tokenize(clean_text)    # using the tokenizer to tokenize the cleaned text
tweet_data['tweet'] = tweet_data['tweet'].apply(remove_hashsymbols) # applying the function to the tweet column

# Removing terms with a length of 1
tweet_data['tweet'] = tweet_data['tweet'].apply(lambda x: [item for item in x if len(item) > 1])

# Pritning the first 5 preprocessed tweets
print('The first 5 preprocessed tweets are:\n',tweet_data.head())

# Top 10 most frequent terms in the tweets
results = Counter()
tweet_data['tweet'].apply(results.update)
#print the top 10 most common terms in the tweet 
print('The top 10 most frequent terms in the tweets and their counts are:\n',results.most_common(10))

# Data formatting for predictive modeling
# Convert the tokenized tweets into a single string
tweet_data['tweet'] = tweet_data['tweet'].apply(lambda x: ' '.join(x))

# Split the data into training and testing sets
X = tweet_data['tweet']
y = tweet_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # 80% training and 20% testing

# TF-IDF values for the terms as a feature to get into a vector space model
vectorizer = TfidfVectorizer(max_features=5000)    # max_features = 5000 to limit the number of features to build the vocabulary
X_train_tfidf = vectorizer.fit_transform(X_train)   # fit and transform the training data
X_test_tfidf = vectorizer.transform(X_test)         # transform the testing data

# Model Building: Ordinary Logistic Regression Model 
logreg = LogisticRegression()
logreg.fit(X_train_tfidf, y_train)

# Model Evaluation : Accuracy, recall, and f_1 score using sklearn classification_report and accuracy_score
y_pred = logreg.predict(X_test_tfidf)
print('The accuracy score for Logistic Regression on original dataset is:',accuracy_score(y_test, y_pred))
print('The classification report for Logistic Regression on original dataset is:\n',classification_report(y_test, y_pred))

# Plot the confusion matrix for Logistic Regression using seaborn and matplotlib to find the true positive, true negative, false positive, and false negative of the model
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
acc_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, y_pred)*100)
plt.title('Confusion Matrix for Logistic Regression on original dataset \n '+acc_title, size = 15)
plt.show()

# Since the f1 score is low for minority class shown in classificqation report and confusion matrix, implying class imbalance
# This can be further confimed by the countplot of the target variable
# Checking for imbalances in target by checking total class count
Non_Hate_percent =  round(tweet_data['label'].value_counts()[0]/len(tweet_data) * 100,2)
Hate_percent =  round(tweet_data['label'].value_counts()[1]/len(tweet_data) * 100,2)

# Plotting the target variable to check for imbalances
sns.countplot(x='label', data=tweet_data)
plt.title('Tweet Label Count \n (Label 0 = Non-Hate is %f percent of the dataset || Label 1 = Hate is %f percent of the dataset' % (Non_Hate_percent, Hate_percent) ,fontsize=10)
plt.xticks(range(2), ["Non-Hate", "Hate"])
plt.show()

# Since the target variable is imbalanced for minority, we can use SMOTEENN to balance the target variable
# Applying Oversampling and then Undersampling on training dataset to balance the target variable using SMOTEENN
# SMOTEENN is a hybrid method that applies SMOTE to oversample the minority calss with synthetic new data and then applies Edited Nearest Neighbours to clean the dataset
# Edited Nearest Neighbours is a method that removes any noisy majoirity class samples from the dataset close to minority class samples
from imblearn.combine import SMOTEENN
nm3 = SMOTEENN(sampling_strategy=0.4) # using SMOTEENN to create an object nm3
x_samples,y_samples= nm3.fit_resample(X_train_tfidf, y_train)    # fit and transform the training data
print(sorted(Counter(y_samples).items()))   # print the count of the target variable after balancing
print('Shape of the new oversmapled and undersampled dataset for features :'  , x_samples.shape, 'and target:',y_samples.shape)     # print the shape of the new dataset
print('Original dataset shape {}'.format(Counter(y_train)))
print('Resampled dataset shape {}'.format(Counter(y_samples)))

# Applying the Logistic Regression model on the balanced dataset using SMOTEENN
logreg_balanced = LogisticRegression()
logreg_balanced.fit(x_samples, y_samples)
# Model Evaluation : Accuracy, recall, and f_1 score.
y_pred2 = logreg_balanced.predict(X_test_tfidf)
print('The accuracy score for Logistic Regression on balanced dataset is:',accuracy_score(y_test, y_pred2))
print('The classification report for Logistic Regression on balanced dataset is:\n',classification_report(y_test, y_pred2))

# Plot the confusion matrix for Logistic Regression with new balanced dataset
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, y_pred2), annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
acc_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, y_pred2)*100)
plt.title('Confusion Matrix for Logistic Regression on balanced dataset \n '+acc_title, size = 15)
plt.show()

# Import GridSearch and StratifiedKFold for hyperparameter tuning of the logistic regression model
from sklearn.model_selection import GridSearchCV, StratifiedKFold
logreg_cv = LogisticRegression(class_weight='balanced', max_iter=2000) # This time we use class_weight='balanced' to account for the class imbalance in the original dataset
# Instantiate the GridSearchCV object and run the search for the best parameters for different solvers, regularization, and C values for logistic regression.
# We also use StratifiedKFold to ensure that the class distribution is preserved in the training and testing datasets for each fold. 
# StratifiedKFold is a cross-validation object that is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
best_parameters = GridSearchCV(logreg_cv, param_grid={'C': np.linspace(0.01, 100, 100), 'penalty': ['l2', 'l1'], 'solver': ['lbfgs','liblinear']},
                               scoring='recall', cv=StratifiedKFold(n_splits=4), verbose=0, n_jobs=-1, error_score=0)  # using recall as the scoring metric for the best parameters
best_parameters.fit(X_train_tfidf, y_train)  # fit the model to the training data, however an error will be thrown as solver='lbfgs' does not support L1 regularization which can be ingnored.

# Print the best tuned parameters that give the recall best score
print("\n Best Tuned Logistic Regression Parameters: {}".format(best_parameters.best_params_))
print("Best score is {}".format(best_parameters.best_score_))

# Extract the best parameters for logistic regression model for the best recall score from the grid search
paramC = best_parameters.best_params_.get('C')
penalty = best_parameters.best_params_.get('penalty')
solver = best_parameters.best_params_.get('solver')

# Finally predicting using best parameters for logistic regression  model and also employing custom class weights
# Custom class weights are used to account for the class imbalance in the original dataset. 
# They are used to assign different weights to the classes such that the model learns from the minority class as well.
print('\n Ratio of class 0 to class 1 that will be used for class weights:', len(y_train[y_train==0])/len(y_train[y_train==1]))
# The ratio of class 0 to class 1 is 13:1, so we use class weights of 1:13 for class 0 and class 1 respectively.
weights = {0: 1, 1: 13}
logreg_best = LogisticRegression(class_weight=weights, max_iter=1000, C=paramC, penalty=penalty, solver=solver)
logreg_best.fit(X_train_tfidf, y_train)

# Model Evaluation with best parameters: Accuracy, recall, and f_1 score.
y_pred3 = logreg_best.predict(X_test_tfidf)
print('\n The accuracy score for Logistic Regression for best parameters is:',accuracy_score(y_test, y_pred3))
print('\n The recall score for Logistic Regression for best parameters is:',recall_score(y_test, y_pred3))
print('The f1 score for Logistic Regression for best parameters is:',f1_score(y_test, y_pred3))
print('The classification report for Logistic Regression for best parameters is:\n',classification_report(y_test, y_pred3))

# Plot the confusion matrix for Logistic Regression with best parameters
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, y_pred3), annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
acc_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, y_pred3)*100)
plt.title('Confusion Matrix for Logistic Regression with best parameters \n '+acc_title, size = 15)
plt.show()


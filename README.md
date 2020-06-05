# TopicModelAUD

## Introduction
Craigslist makes money through only a few revenue streams. Most of the postings are free and there are only few avenues where they charge to post a job listing or apartment listing. This is true for only a few major cities. However, Craigslist’s aim is to provide the simplest functionality in the service of society.
1.	Situation and Complication- In order to help boost number of users and visit time, we propose to use the existing ‘Featured Listings’ section. In the current situation any listing gets into featured section based on the user likes. We propose to use this list of posts as a proxy for viral posts and predict the virality of a new post.
2.	Solution- We plan to use machine learnings algorithm to classify posts as viral/not viral. This will help to increase customer time on site and bring in new customers. We also plan to create a separate section for Obscene posts which might hurt the user satisfaction for a few users.
3.	Impact- This functionality would make it easier for website users to identify interesting and amusing posts. This move will potentially bring new users as people share posts with their friends. 


## Data Analysis
Our analysis process can be broken down into the following 7 stages.
1.	Fetching unstructured data from the website - We scraped posts from the ‘Best of Craigslist’ page. This makes up our 1’s (featured listings) in the dataset. We scraped posts from craigslist in different categories. These are our 0’s (non-featured listings) in the dataset
2.	Preprocessing - We processed the data to remove any junk information from the dataset such as null values and insignificant variables
3.	Sampling - We found that our target variable (featured/non-featured) is unbalanced with only ~2% of featured posts. So, we performed random down-sampling to balance the dataset
4.	Creating term document matrix - We brought the text data into a format which our machine learning models can understand. We tokenized, lemmatized, removed stop-words and transformed the text into tf-idf matrix. This increased the size of our dataset exponentially. To further make our models simpler for faster processing, we processed the data to curb near-zero variables 
5.	Final dataset – We split the dataset into 80:20 ratio for training and testing our models respectively
6.	Model creation – We tried various classifiers to test which would be flexible enough to fit on the training dataset and at the same time can be generalized for different datasets. The models are listed below – <br />
a.	Logistic Regression (LR) <br />
b.	Support Vector Machines (SVM) <br />
c.	Random Forest Classifier (RF) <br />
d.	K-Nearest Neighbor (KNN) <br />
e.	Gaussian Naive Bayes (GNB) <br />
f.	Decision Tree Classifier (DT) <br />
g.	Gradient Boosting Classifier (GBC) <br />
h.	Adaboost Classifier (ADC) <br />
i.	MLP Classifier (DL) <br />
j.	LSTM <br />
7.	Fine Tuning – Our initial 10-fold cross-validation results showed that LSTM, deep learning, gradient boosting and random forest classifiers perform best with a maximum accuracy of ~91%. We further tried to improve these models by optimizing the model parameters using grid search. The model parameters and results post optimization are given below – <br />
a.	LSTM Best Score: 92.6 and Best Parameters: {‘embedding_dim': 100, 'activation': ‘softmax, optimizer: ‘adam’, 'loss':’ categorical_crossentropy’, 'dropout':0.2, 'recurrent_dropout':0.2} <br />
b.	DL Best Score: 90.3 and Best Parameters: {‘hidden_layer_sizes ': (2,2,4), 'activation': ‘relu’, solver': ‘adam’, 'alpha': 0.05, 'learning_rate': ‘adaptive’} <br />
c.	GBC Best Score: 90.1 and Best Parameters: {'learning_rate': 0.05, 'max_depth': 8, 'max_features': 0.3, 'min_samples_split': 3, 'random_state': 1234} <br />
d.	RF Best Score: 89.9 and Best Parameters: {'criterion': 'gini', 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 30, 'random_state': 44} <br />

## Results
![image](https://user-images.githubusercontent.com/8051156/77882491-0ae48800-722f-11ea-97a8-3043579d78e9.png)

![image](https://user-images.githubusercontent.com/8051156/77882268-a4f80080-722e-11ea-8f31-362a9852db93.png)

![image](https://user-images.githubusercontent.com/8051156/77882371-dc66ad00-722e-11ea-8c71-acf7852b57b2.png)



#!/usr/bin/env python
# coding: utf-8

# # Loading the twitter essentials 

# In[ ]:


Consumer_key = 'VzNRiotxgyKZxS2JPBSfMBnvB'
Consumer_secret = 'Q0e15B8g2ggc68UsZypQLFf93W9z4OarU0bAlfxYkh68mNk16j'
Access_token = '46323541-HOHPe8KuCvKaHQJhVDUpnryCl0RvLWmOXN9K8H46g'
Access_secret = 'bYUgXTYR3W1gVJCHOTbTfiLY4FF5Wibbkj36TNke2wUDf'


# ## Load required package

# In[13]:


import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pandas as pd
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas_profiling

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.utils import resample


# ## Create a class function to store Twitter data

# In[ ]:


class TwitterClient(object): 
    ''' 
    Generic Twitter Class for sentiment analysis. 
    '''
    def __init__(self): 
        ''' 
        Class constructor or initialization method. 
        '''
        # keys and tokens from the Twitter Dev Console 
        consumer_key = Consumer_key
        consumer_secret = Consumer_secret
        access_token = Access_token
        access_token_secret = Access_secret
  
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 
  
    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 
  
    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'
  
    def get_tweets(self, query, count = 10): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 
  
        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.api.search(q = query, count = count) 
  
            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 
  
                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 
  
                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 
  
            # return parsed tweets 
            return tweets 
  
        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e)) 
  


# ## Pull Craigslist review from Twitter

# In[ ]:


api = TwitterClient() 
# calling function to get tweets 
tweets = api.get_tweets(query = 'Craigslist', count = 20000000) 
tweets_2 = api.get_tweets(query = "craigslist's", count = 20000000) 


# picking positive tweets from tweets 
# ptweets = tweets
n_tweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
n_tweets_2 = [tweet for tweet in tweets_2 if tweet['sentiment'] == 'negative'] 

A =[]
# for tweet in n_tweets: 
#         A.append(tweet['text'])
        
for tweet in n_tweets_2: 
        A.append(tweet['text'])
        
        text = A

text = ' '.join(text)

text2 = text.split()



stopwords = ['Craigslist', 'Craigslist',"Craigslist,",'Craigslist’s','craigslist','Long','listings','RT','yet','RARE',' Craigslist','Craigslist ']
for word in list(text2):  # iterating on a copy since removing will mess things up
    if word in stopwords:
        text2.remove(word)
#print(text2)

import tldextract
text2 = [tldextract.extract(s).domain for s in text2]

text2 = ' '.join(text2)
#print(len(text2))

wordcloud = WordCloud().generate(text2)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Comparison of Negative reviews of the companies

# ### Pull Craigslist Negative reviews

# In[ ]:


def main(): 
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets(query = "Craigslist's", count = 2000000) 

    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    
    # percentage of positive tweets 
    print("Negative tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
  
if __name__ == "__main__": 
    # calling main function 
    main() 


# ### Pull Ebay Negative reviews

# In[ ]:


def main(): 
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets(query = 'Ebay', count = 2000000) 

    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    
    # percentage of positive tweets 
    print("Negative tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
  
if __name__ == "__main__": 
    # calling main function 
    main() 


# ### Pull Amazon Negative reviews

# In[ ]:


def main(): 
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets(query = 'Amazon', count = 2000000) 

    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    
    # percentage of positive tweets 
    print("Negative tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
  
if __name__ == "__main__": 
    # calling main function 
    main() 


# ### Pull BestBuy Negative reviews

# In[ ]:


def main(): 
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets(query = 'BestBuy', count = 2000000) 

    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    
    # percentage of positive tweets 
    print("Negative tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
  
if __name__ == "__main__": 
    # calling main function 
    main() 


# ### Pull Walmart Negative reviews

# In[ ]:


def main(): 
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets(query = 'Walmart', count = 2000000) 

    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    
    # percentage of positive tweets 
    print("Negative tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
  
if __name__ == "__main__": 
    # calling main function 
    main() 


# # Solution 1 - NSFW post flagging

# In[16]:


# Read the posts scraped from the scrapy spider.
data_sale = pd.read_csv('for_sale.csv')
data_comm = pd.read_csv('community.csv')
data_sale['label'] = 0
data_comm['label'] = 0
data_sale['source'] = 'community'
data_comm['source'] = 'for sale'
del data_comm['location']
del data_sale['location']
data2 = pd.read_csv('bestposts.csv')
data1 = data_sale.append(data_comm)


# In[17]:


data2['source'] = "best post"
data2['Label'] = 1
data2['Price'] = -999


# In[18]:


data2 = data2.reindex(columns=['date', 'title', 'title_link', 'region', 'Price', 'depth',
        'download_timeout', 'download_slot', 'download_latency', 'Text',
        'category', 'Label', 'source'])


# In[19]:


data = data1.append(data2)


# In[20]:


df = data.copy(deep = True)
df_sol1= data.copy(deep = True)
#df = pd.DataFrame(df.head(1000))


# In[21]:


# Text cleaning
df_sol1['y'] = df_sol1['source'].apply(lambda x: 1 if x == "best post" else 0)

list_keep= ['Text']
df_sol1 = df_sol1[list_keep]

# 18+ list
list_stop_slang = ['xxx','xrated','willy','willies','whore','whoar','wanky','wanker','wank','wang','w00se','vulva','viagra','vagina','v1gra','v14gra','twunter','twunt','twatty','twathead','twat','tw4t','turd','tosser','titwank','tittywank','tittyfuck','titties','tittiefucker','tittie5','titt','tits','titfuck','tit','testicle','testical','teez','teets','t1tties','t1tt1e5','s_h_i_t','spunk','spac','son-of-a-bitch','snatch','smut','smegma','sluts','slut','skank','shitty','shittings','shitting','shitters','shitter','shitted','shits','shitings','shiting','shithead','shitfull','shitfuck','shitey','shited','shite','shitdick','shit','shi+','shemale','shagging','shaggin','shagger','shag','sh1t','sh!t','sh!+','sex','semen','scrotum','scrote','scroat','screwing','schlong','sadist','s.o.b.','s hit','rimming','rimjaw','retard','rectum','pussys','pussy','pussies','pussi','pusse','pube','pron','pricks','prick','pornos','pornography','porno','porn','poop','pissoff','pissing','pissin','pissflaps','pisses','pissers','pisser','pissed','piss','pimpis','pigfucker','phuq','phuks','phukking','phukked','phuking','phuked','phuk','phuck','phonesex','penisfucker','penis','pecker','pawn','p0rn','orgasms','orgasm','orgasims','orgasim','nutsack','numbnuts','nobjokey','nobjocky','nobhead','nob jokey','nob','niggers','nigger','niggaz','niggas','niggah','nigga','nigg4h','nigg3r','nazi','n1gger','n1gga','mutherfucker','muther','muthafuckker','muthafecker','mutha','muff','motherfucks','motherfuckka','motherfuckings','motherfucking','motherfuckin','motherfuckers','motherfucker','motherfucked','motherfuck','mother fucker','mothafucks','mothafuckings','mothafucking','mothafuckin','mothafuckers','mothafucker','mothafucked','mothafuckaz','mothafuckas','mothafucka','mothafuck','mofo','mof0','mo-fo','masturbate','masterbations','masterbation','masterbate','masterbat3','masterbat*','masterb8','master-bate','masochist','ma5terbate','ma5terb8','m45terbate','m0fo','m0f0','lusting','lust','lmfao','labia','l3itch','l3i+ch','kunilingus','kums','kumming','kummer','kum','kondums','kondum','kock','knobjokey','knobjocky','knobhead','knobend','knobed','knobead','knob','kawk','jizz','jizm','jiz','jism','jerk-off','jap','jackoff','jack-off','hotsex','horny','horniest','hore','homo','hoer','hoare','hoar','heshe','hell','hardcoresex','goddamned','goddamn','god-damned','god-dam','God','goatse','gaysex','gaylord','gangbangs','gangbanged','gangbang','f_u_c_k','fux0r','fux','fukwit','fukwhit','fuks','fukkin','fukker','fuker','fuk','fudgepacker','fudge packer','fuckwit','fuckwhit','fucks','fuckme','fuckingshitmotherfucker','fuckings','fucking','fuckin','fuckheads','fuckhead','fuckers','fucker','fucked','fucka','fuck','fooker','fook','flange','fistfucks','fistfuckings','fistfucking','fistfuckers','fistfucker','fistfucked','fistfuck','fingerfucks','fingerfucking','fingerfuckers','fingerfucker','fingerfucked','fingerfuck','fellatio','fellate','felching','fecker','feck','fcuking','fcuker','fcuk','fatass','fanyy','fannyfucker','fannyflaps','fanny','fags','fagots','fagot','faggs','faggot','faggitt','fagging','fag','f4nny','f u c k e r','f u c k','ejakulate','ejaculation','ejaculatings','ejaculating','ejaculates','ejaculated','ejaculate','dyke','duche','doosh','donkeyribber','dogging','doggin','dog-fucker','dlck','dirsa','dinks','dink','dildos','dildo','dickhead','dick','damn','d1ck','cyberfucking','cyberfuckers','cyberfucker','cyberfucked','cyberfuck','cyberfuc','cyalis','cunts','cuntlicking','cuntlicker','cuntlick','cunt','cunnilingus','cunillingus','cunilingus','cumshot','cums','cumming','cummer','cum','crap','cox','coon','coksucka','cokmuncher','cok','cocksukka','cocksuka','cocksucks','cocksucking','cocksucker','cocksucked','cocksuck','cocks','cockmuncher','cockmunch','cockhead','cockface','cock-sucker','cock','cnut','clits','clitoris','clit','cl1t','cipa','chink','cawk','carpet muncher','c0cksucker','c0ck','buttplug','buttmuch','butthole','butt','bunny fucker','bum','bugger','buceta','breasts','booooooobs','booooobs','boooobs','booobs','boobs','boob','boner','bollok','bollock','boiolas','blowjobs','blowjob','blow job','bloody','bitching','bitchin','bitches','bitchers','bitcher','bitch','biatch','bi+ch','bestiality','bestial','bellend','beastiality','beastial','bastard','ballsack','balls','ballbag','b1tch','b17ch','b00bs','b!tch','a_s_s','asswhole','assholes','asshole','assfukka','assfucker','asses','ass-fucker','ass','arse','arrse','ar5e','anus','anal','a55','5hit','5h1t','4r5e']
pat = '|'.join([r'\b{}\b'.format(w) for w in list_stop_slang])

df_sol1 = df_sol1.assign(new=df_sol1.replace(dict(Text={pat: ''}), regex=True))

sentences = pd.Series(df_sol1.Text)
sentences2 = pd.Series(df_sol1.new)

# remove anything but characters and spaces
sentences = sentences.str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()

splitwords = [ nltk.word_tokenize( str(sentence) ) for sentence in sentences ]

wordcounts = [ len(words) for words in splitwords ]
#print(wordcounts)
sentences2 = sentences2.str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()

splitwords2 = [ nltk.word_tokenize( str(sentence) ) for sentence in sentences2 ]

wordcounts2 = [ len(words) for words in splitwords2 ]
#print(wordcounts2)

df_sol1['count_text'] = wordcounts
df_sol1['count_new'] = wordcounts2
df_sol1['flag'] = df_sol1['count_text'] - df_sol1['count_new']


# In[22]:


df_sol1_mature = df_sol1[df_sol1['flag']!=0]
df_sol1_clean = df_sol1[df_sol1['flag']==0]


# In[23]:


count = 0
for i in range(len(wordcounts2)-1):
    if wordcounts2[i] == wordcounts[i]:
        count = count + 1
    else:
        count = count
A = 1- count/ len(wordcounts2)
print(str(round(A*100,0))+"% percent of the review contains the Less than 18 content")


# # Solution 2 - Viral post prediction

# In[ ]:


df['y'] = df['source'].apply(lambda x: 1 if x == "best post" else 0)

# Keep only description and target variable
list_keep= ['Text','y']
df = df[list_keep]
df.Text= df.Text.astype(str)


# ### Random undersampling

# In[ ]:


df_majority = df[df.y==0]
df_minority = df[df.y==1]

df_majority_undersampled = df_majority.sample(df_minority.shape[0])

df_final = pd.concat([df_majority_undersampled, df_minority], axis=0)


# ### Creating term document matrix

# In[ ]:


list_df = df_final.Text.tolist()

list_tokenize = []
list_lematize = []
list_stop_remove = []

lemmatizer = nltk.stem.WordNetLemmatizer()

for i in list_df:
    #TOKENIZE
    token_d1 = nltk.word_tokenize(i.lower())
    list_tokenize.append(token_d1)
    
    #LEMMATIZE
    token_d2 = token_d1    
    lemmatized_token_d2 = [lemmatizer.lemmatize(token) for token in token_d2 if token.isalpha()]
    list_lematize.append(lemmatized_token_d2)
    
    #STOP WORDS REMOVAL
    stop_words_removed = [token for token in token_d2 if not token in stopwords.words('english') if token.isalpha()]
    list_stop_remove.append(stop_words_removed) 


def func_def(doc):
    return doc

tf_define = TfidfVectorizer(analyzer='word', tokenizer=func_def, preprocessor=func_def,token_pattern=None,
    ngram_range=(1, 2), 
    min_df=5)  

tf_define.fit(list_stop_remove)
v = tf_define.transform(list_stop_remove)
terms = tf_define.get_feature_names()


# ### Further processing

# In[ ]:


# Convert sparse matrix to dataframe
Q = pd.DataFrame(v.toarray())


# In[ ]:


cols = []
for k in tf_define.vocabulary_:
    cols.append(k)
rows = pd.DataFrame(Q)
rows.columns = cols


# In[ ]:


rows.shape, df_final.shape


# In[ ]:


df_final.reset_index(drop=True, inplace=True)


# In[ ]:


min_number_of_non_zeros = df_final.shape[0]*0.05


# In[ ]:


col = []
for name, values in rows.iteritems():
    if (len(rows[name].nonzero()[0])<=min_number_of_non_zeros):
        col.append(name)


# In[ ]:


df_X = rows.drop(col,axis=1)


# In[ ]:


df_X.shape


# ### Final dataset

# In[ ]:


final_dataset = pd.concat([df_final.y, df_X], axis=1)


# In[ ]:


final_dataset.shape, df_final.shape


# In[ ]:


# final_dataset.to_csv("AUD_undersampled_final_dataset.csv",index=False)
# df_final.to_csv("AUD_undersampled_raw.csv",index=False)


# In[ ]:


# final_dataset = pd.read_csv("AUD_undersampled_final_dataset.csv")
# df_X = final_dataset.drop('y',axis=1)


# In[ ]:


# Test train split
X_train, X_test, y_train, y_test = train_test_split(df_X, final_dataset['y'], test_size=0.2, random_state=1234)


# In[ ]:


X_train.shape, y_train.shape


# ### Model Creation

# In[ ]:


seed = 1234
"""Building machine learning models: 
We will try 10 different classifiers to find the best classifier after tunning model's hyperparameters that will best generalize the unseen(test) data."""

'''Now initialize all the classifiers object.'''
'''#1.Logistic Regression'''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

'''#2.Support Vector Machines'''
from sklearn.svm import SVC
svc = SVC(gamma = 'auto')

'''#3.Random Forest Classifier'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = seed, n_estimators = 100)

'''#4.KNN'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

'''#5.Gaussian Naive Bayes'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

'''#6.Decision Tree Classifier'''
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = seed)

'''#7.Gradient Boosting Classifier'''
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state = seed)

'''#8.Adaboost Classifier'''
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(random_state = seed)

'''#9.ExtraTrees Classifier'''
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(random_state = seed)

'''#10.MLP Classifier'''
from sklearn.neural_network import MLPClassifier
dl = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10), random_state=seed, max_iter=100)


# In[ ]:


def train_accuracy(model):
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    train_accuracy = np.round(train_accuracy*100, 2)
    return train_accuracy


'''Models with best training accuracy:'''
train_accuracy = pd.DataFrame({'Train_accuracy(%)':[train_accuracy(lr), 
                                                    train_accuracy(svc), 
                                                    train_accuracy(rf), 
                                                    train_accuracy(knn), 
                                                    train_accuracy(gnb), 
                                                    train_accuracy(dt), 
                                                    train_accuracy(gbc), 
                                                    train_accuracy(abc), 
                                                    train_accuracy(etc),
                                                    train_accuracy(dl)
                                                   ]})
train_accuracy.index = ['LR', 'SVC', 'RF', 'KNN', 'GNB', 'DT', 'GBC', 'ABC', 'ETC','DL']
sorted_train_accuracy = train_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)
print('**Training Accuracy of the Classifiers:**')
print(sorted_train_accuracy)


# In[ ]:


'''Create a function that returns mean cross validation score for different models.'''
def x_val_score(model):
    from sklearn.model_selection import cross_val_score
    x_val_score = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy').mean()
    x_val_score = np.round(x_val_score*100, 2)
    return x_val_score

"""Let's perform k-fold (k=10) cross validation to find the classifier with the best cross validation accuracy."""
x_val_score = pd.DataFrame({'X_val_score(%)':[x_val_score(lr), 
                                              x_val_score(svc), 
                                              x_val_score(rf), 
                                              x_val_score(knn), 
                                              x_val_score(gnb), 
                                              x_val_score(dt), 
                                              x_val_score(gbc), 
                                              x_val_score(abc), 
                                              x_val_score(etc),
                                              x_val_score(dl)
                                             ]})
x_val_score.index = ['LR', 'SVC', 'RF', 'KNN', 'GNB', 'DT', 'GBC', 'ABC', 'ETC', 'DL']
sorted_x_val_score = x_val_score.sort_values(by = 'X_val_score(%)', ascending = False) 
print('**Models 10-fold Cross Validation Score:**')
print(sorted_x_val_score)


# In[ ]:


'''Make prediction using all the trained models.'''
model_prediction = pd.DataFrame({'RF':rf.predict(X_test), 'GBC':gbc.predict(X_test), 'ABC':abc.predict(X_test),
                                 'ETC':etc.predict(X_test), 'DT':dt.predict(X_test), 'SVC':svc.predict(X_test), 
                                 'KNN':knn.predict(X_test), 'LR':lr.predict(X_test), 'DL':dl.predict(X_test)})

"""Let's see how each model classifies a prticular class."""
print('**All the Models Prediction:**')
print(model_prediction.head())


# In[ ]:


'''All the Models AUC score on test before optimization.'''
from sklearn.metrics import roc_auc_score
print('**All the Models AUC score on test:**')
for k,v in model_prediction.items():
    print(k,"\t",roc_auc_score(y_test,v))


# In[ ]:


"""Define all the models' hyperparameters one by one first::"""

'''Define hyperparameters the logistic regression will be tuned with. For LR, the following hyperparameters are usually tunned.'''
lr_params = {'penalty':['l1', 'l2'],
             'C': np.logspace(0, 4, 10)}

'''For GBC, the following hyperparameters are usually tunned.'''
gbc_params = {'learning_rate': [0.01, 0.02, 0.05, 0.01],
              'max_depth': [4, 6, 8],
              'max_features': [1.0, 0.3, 0.1], 
              'min_samples_split': [ 2, 3, 4],
              'random_state':[seed]}

'''For SVC, the following hyperparameters are usually tunned.'''
svc_params = {'C': [6, 7, 8, 9, 10, 11, 12], 
              'kernel': ['linear','rbf'],
              'gamma': [0.5, 0.2, 0.1, 0.001, 0.0001]}

'''For DT, the following hyperparameters are usually tunned.'''
dt_params = {'max_features': ['auto', 'sqrt', 'log2'],
             'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
             'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             'random_state':[seed]}

'''For RF, the following hyperparameters are usually tunned.'''
rf_params = {'criterion':['gini','entropy'],
             'n_estimators':[10, 15, 20, 25, 30],
             'min_samples_leaf':[1, 2, 3],
             'min_samples_split':[3, 4, 5, 6, 7], 
             'max_features':['sqrt', 'auto', 'log2'],
             'random_state':[44]}

'''For KNN, the following hyperparameters are usually tunned.'''
knn_params = {'n_neighbors':[3, 4, 5, 6, 7, 8],
              'leaf_size':[1, 2, 3, 5],
              'weights':['uniform', 'distance'],
              'algorithm':['auto', 'ball_tree','kd_tree','brute']}

'''For ABC, the following hyperparameters are usually tunned.'''
abc_params = {'n_estimators':[1, 5, 10, 15, 20, 25, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300],
              'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],
              'random_state':[seed]}

'''For ETC, the following hyperparameters are usually tunned.'''
etc_params = {'max_depth':[None],
              'max_features':[1, 3, 10],
              'min_samples_split':[2, 3, 10],
              'min_samples_leaf':[1, 3, 10],
              'bootstrap':[False],
              'n_estimators':[100, 300],
              'criterion':["gini"], 
              'random_state':[seed]}

'''For DL, the following hyperparameters are usually tunned.'''
dl_params = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}


# In[ ]:


'''Create a function to tune hyperparameters of the selected models.'''
def tune_hyperparameters(model, params):
    from sklearn.model_selection import GridSearchCV
    global best_params, best_score
    # Construct grid search object with 10 fold cross validation.
    grid = GridSearchCV(model, params, verbose = 0, cv = 10, scoring = 'accuracy', n_jobs = -1)
    # Fit using grid search.
    grid.fit(X_train, y_train)
    best_params, best_score = grid.best_params_, np.round(grid.best_score_*100, 2)
    return best_params, best_score


# In[ ]:


# '''Tune LR hyperparameters.'''
# tune_hyperparameters(lr, params = lr_params)
# lr_best_params, lr_best_score = best_params, best_score
# print('aLR Best Score:', lr_best_score)
# print('And Best Parameters:', lr_best_params)


# In[ ]:


# """Tune GBC's hyperparameters."""
# tune_hyperparameters(gbc, params = gbc_params)
# gbc_best_score, gbc_best_params = best_score, best_params
# print('GBC Best Score:', gbc_best_score)
# print('And Best Parameters:', gbc_best_params)


# In[ ]:


# """Tune SVC's hyperparameters."""
# tune_hyperparameters(svc, params = svc_params)
# svc_best_score, svc_best_params = best_score, best_params
# print('SVC Best Score:', svc_best_score)
# print('And Best Parameters:', svc_best_params)


# In[ ]:


# """Tune DT's hyperparameters."""
# tune_hyperparameters(dt, params = dt_params)
# dt_best_score, dt_best_params = best_score, best_params
# print('DT Best Score:', dt_best_score)
# print('And Best Parameters:', dt_best_params)


# In[ ]:


# """Tune RF's hyperparameters."""
# tune_hyperparameters(rf, params = rf_params)
# rf_best_score, rf_best_params = best_score, best_params
# print('RF Best Score:', rf_best_score)
# print('And Best Parameters:', rf_best_params)


# In[ ]:


# """Tune KNN's hyperparameters."""
# tune_hyperparameters(knn, params = knn_params)
# knn_best_score, knn_best_params = best_score, best_params
# print('KNN Best Score:', knn_best_score)
# print('And Best Parameters:', knn_best_params)


# In[ ]:


# """Tune ABC's hyperparameters."""
# tune_hyperparameters(abc, params = abc_params)
# abc_best_score, abc_best_params = best_score, best_params
# print('ABC Best Score:', abc_best_score)
# print('And Best Parameters:', abc_best_params)


# In[ ]:


# """Tune ETC's hyperparameters."""
# tune_hyperparameters(etc, params = etc_params)
# etc_best_score, etc_best_params = best_score, best_params
# print('ETC Best Score:', etc_best_score)
# print('And Best Parameters:', etc_best_params)


# In[ ]:


# """Tune DL's hyperparameters."""
# tune_hyperparameters(dl, params = dl_params)
# dl_best_score, etc_best_params = best_score, best_params
# print('ETC Best Score:', etc_best_score)
# print('And Best Parameters:', etc_best_params)


# In[ ]:


# '''Create a dataframe of tunned scores and sort them in descending order.'''
# tunned_scores = pd.DataFrame({'Tunned_accuracy(%)': [lr_best_score, gbc_best_score, svc_best_score, dt_best_score, rf_best_score, knn_best_score, abc_best_score, etc_best_score, dl_best_score]})
# tunned_scores.index = ['LR', 'GBC', 'SVC', 'DT', 'RF', 'KNN', 'ABC', 'ETC', 'DL']
# sorted_tunned_scores = tunned_scores.sort_values(by = 'Tunned_accuracy(%)', ascending = False)
# print('**Models Accuracy after Optimization:**')
# print(sorted_tunned_scores)


# In[ ]:


'''#4.Create a function that compares cross validation scores with tunned scores for different models by plotting them.'''
def compare_scores(accuracy):
    global ax1   
    font_size = 15
    title_size = 18
    ax1 = accuracy.plot.bar(legend = False,  title = 'Models %s' % ''.join(list(accuracy.columns)), figsize = (18, 5), color = 'sandybrown')
    ax1.title.set_size(fontsize = title_size)
    # Removes square brackets and quotes from column name after to converting list.
    pct_bar_labels()
    plt.ylabel('% Accuracy', fontsize = font_size)
    plt.show()

'''Compare cross validation scores with tunned scores to find the best model.'''
print('**Comparing Cross Validation Scores with Optimized Scores:**')
print(sorted_x_val_score)
print(sorted_tunned_scores)


# In[ ]:


'''Instantiate the models with optimized hyperparameters.'''
rf  = RandomForestClassifier(**rf_best_params)
gbc = GradientBoostingClassifier(**gbc_best_params)
svc = SVC(**svc_best_params)
knn = KNeighborsClassifier(**knn_best_params)
etc = ExtraTreesClassifier(**etc_best_params)
lr  = LogisticRegression(**lr_best_params)
dt  = DecisionTreeClassifier(**dt_best_params)
abc = AdaBoostClassifier(**abc_best_params)
dl = MLPClassifier(**dl_best_params)

'''Train all the models with optimised hyperparameters.'''
models = {'RF':rf, 'GBC':gbc, 'SVC':svc, 'KNN':knn, 'ETC':etc, 'LR':lr, 'DT':dt, 'ABC':abc, 'DL':dl}
print('**10-fold Cross Validation after Optimization:**')
score = []
for x, (keys, items) in enumerate(models.items()):
    # Train the models with optimized parameters using cross validation.
    # No need to fit the data. cross_val_score does that for us.
    # But we need to fit train data for prediction in the follow session.
    from sklearn.model_selection import cross_val_score
    items.fit(X_train, y_train)
    scores = cross_val_score(items, X_train, y_train, cv = 10, scoring = 'accuracy')*100
    score.append(scores.mean())
    print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), keys))


# In[ ]:


# '''Stack up all the models above, optimized using xgboost'''
# from mlxtend.classifier import StackingCVClassifier
# stack_gen = StackingCVClassifier(classifiers=(DL,RF,LR,GBC),
#                                 meta_classifier=DL,
#                                 use_features_in_secondary=True)
# stack_gen.fit(np.array(X_train), np.array(y_train))


# In[ ]:


# from sklearn.metrics import accuracy_score
# list_val = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# acc_save = []

# for i in list_val:
#     from sklearn.neural_network import MLPClassifier
#     DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,i), random_state=1)
#     # training
#     DLmodel.fit(X_train, y_train)
#     y_pred_DL= DLmodel.predict(X_test)
#     # evaluation
#     acc_DL = accuracy_score(y_test, y_pred_DL)
#     acc_save.append(acc_DL)
#     print("Number of nuerons in 2nd layer: {}".format(i),"DL model Accuracy: {:.2f}%".format(acc_DL*100))


# In[ ]:


'''Make prediction using all the trained models.'''
model_prediction = pd.DataFrame({'RF':rf.predict(X_test), 'GBC':gbc.predict(X_test), 'ABC':abc.predict(X_test),
                                 'ETC':etc.predict(X_test), 'DT':dt.predict(X_test), 'SVC':svc.predict(X_test), 
                                 'KNN':knn.predict(X_test), 'LR':lr.predict(X_test), 'DL':DLmodel.predict(X_test)})

"""Let's see how each model classifies a prticular class."""
print('**All the Models Prediction:**')
print(model_prediction.head())


# In[ ]:


'''All the Models AUC score on test after optimization.'''
from sklearn.metrics import roc_auc_score
print('**All the Models AUC score on test:**')
for k,v in model_prediction.items():
    print(k,"\t",roc_auc_score(y_test,v))


# In[ ]:


import matplotlib.pyplot as plt
'''Create a function that plot feature importance by the selected tree based models.'''
def feature_importance(model):
    importance = pd.DataFrame({'Feature': X_train.columns,
                              'Importance': np.round(model.feature_importances_,3)})
    importance = importance.sort_values(by = 'Importance', ascending = False).set_index('Feature')
    return importance

'''Create subplots of feature impotance of rf, gbc, dt, etc, and abc.'''
fig, axes = plt.subplots(3,2, figsize = (20,40))
fig.suptitle('Tree Based Models Feature Importance', fontsize = 28)
tree_models = [rf, gbc, dt, etc, abc]
tree_names = ['RF', 'GBC', 'DT', 'ETC', 'ABC']

for ax, model, name in zip(axes.flatten(), tree_models, tree_names):
    feature_importance(model).plot.barh(ax = ax, title = name, fontsize = 16, color = 'green')
fig.delaxes(ax = axes[2,1]) # We don't need the last subplot.
fig.tight_layout(rect = [0, 0.03, 1, 0.97])


# ### LSTM Classifier

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re


# In[ ]:


df_lstm = df_final
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df_lstm['Text'] = df_lstm['Text'].apply(clean_text)
df_lstm['Text'] = df_lstm['Text'].str.replace('\d+', '')


# In[ ]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


X = tokenizer.texts_to_sequences(df['Text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[ ]:


Y = pd.get_dummies(df['y']).values
print('Shape of label tensor:', Y.shape)


# In[ ]:


XL_train, XL_test, YL_train, YL_test = train_test_split(X,Y, test_size = 0.2, random_state = 1234)
print(XL_train.shape,YL_train.shape)
print(XL_test.shape,YL_test.shape)


# In[ ]:


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[ ]:


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


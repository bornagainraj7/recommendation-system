import pandas as pd
import numpy as np
import pickle
import os


dirname = os.path.dirname(__file__)

logistic_path = os.path.join(dirname, 'pickle', 'logistic_reg_model.pkl')
vectorizer_path = os.path.join(dirname, 'pickle', 'tfidf_vectorizer.pkl')
recommendation_model_path = os.path.join(dirname, 'pickle', 'user_rating.pkl')
mapping_path = os.path.join(dirname, 'pickle', 'prod_id_name_mapping.pkl')
df_path = os.path.join(dirname, 'pickle', 'df.pkl')


logistic = pickle.load(file = open(logistic_path, 'rb'))

vectorizer = pickle.load(file = open(vectorizer_path, 'rb'))

recommendation_model = pickle.load(file = open(recommendation_model_path, 'rb'))

mapping = pickle.load(file = open(mapping_path, 'rb'))

df = pickle.load(file = open(df_path, 'rb'))


def doRecommendations(username):

    tfidf_vectorizer = vectorizer
    lr = logistic
    
    try:
        recommendations = pd.DataFrame(recommendation_model.loc[username]).reset_index()[0 : 50]
    except KeyError:
        errorMessage = f'Hey Mate! we tried hard but couldn\'t find the user "{username}", so we couldn\'t recommend anything \n\
         for "{username}", you can try again by select any of the below username to find their recommendations.'
        print(type(errorMessage))
        return errorMessage, None
    
    recommendations.rename(columns = { recommendations.columns[1]: 'pred_rating' }, inplace = True)
    recommendations = recommendations.sort_values(by = 'pred_rating', ascending = False)[0 : 5]

    recommendations = pd.merge(recommendations, mapping, left_on = 'id', right_on = 'id', how = 'left')

    recommendations = pd.merge(recommendations, df[['id', 'clean_review']], left_on = 'id', right_on = 'id', how = 'left')

    test_data = tfidf_vectorizer.transform(recommendations['clean_review'].values.astype('U'))

    sentiment_pred = lr.predict(test_data)
    sentiment_pred = pd.DataFrame(sentiment_pred,  columns = ['sentiment_predicted'])

    recommendations = pd.concat([recommendations, sentiment_pred], axis = 1)

    groupby = recommendations.groupby('id')

    pred_count_df = pd.DataFrame(groupby['sentiment_predicted'].count()).reset_index()
    pred_count_df.columns = ['id', 'review_count']

    pred_sum_df = pd.DataFrame(groupby['sentiment_predicted'].sum()).reset_index()
    pred_sum_df.columns = ['id', 'pred_pos_review']

    recommendations = pd.merge(pred_count_df, pred_sum_df, left_on = 'id', right_on = 'id', how = 'left')

    recommendations['positive_sentiment_rate'] = round(recommendations.pred_pos_review.div(recommendations.review_count).replace(np.inf, 0) * 100, 2)
    recommendations = recommendations.sort_values(by = 'positive_sentiment_rate', ascending = False)

    recommendations = pd.merge(recommendations, mapping, left_on = 'id', right_on = 'id', how = 'left')

    
    productNameList = recommendations['name'].tolist()
    posSentimentRateList = recommendations['positive_sentiment_rate'].tolist()

    return productNameList, posSentimentRateList

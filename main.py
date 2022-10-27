import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'] )
def logistic_regression():
    if request.method == 'GET':
      return "Hello"
    if request.method == 'POST':
      params = request.get_json(force=True)
      print("Params:  {}".format(params))
      body = dict(params)
      return create_logistic_regression(body)	

def create_logistic_regression(body):
  sentiment = pd.DataFrame(body['sentiment'])
  sentimentDf = pd.get_dummies(sentiment.astype('category'), prefix='Test')
  sentimentDf = check_sentiment_data(sentimentDf)
  starter = [body['close'], body['volume'], body['positive'], body['negative']]
  for i in sentimentDf:
    holder = list(sentimentDf[i])
    starter.append(holder)  
  df = np.array(starter)
  score = {}
  score["next"] = logistic_regression(1,df)
  score["week"] = logistic_regression(10,df)
  score["month"] = logistic_regression(20,df)
  score["ticker"] = body['ticker']
  return score

def check_sentiment_data(df):
  if(len(df.columns) < 4):
    df["add"] = [0] * len(df)
    if(len(df.columns)<4):
      df["plus"] = [0] * len(df)
  return df

def logistic_regression(time, df):
  key = create_key_for_logistic_regression(time, df)
  reshapedDf = df.reshape(len(df[0]),8)
  processedDf = preprocessing.StandardScaler().fit(reshapedDf[:len(reshapedDf)-time,:])
  normalizedDf = processedDf.transform(reshapedDf[:len(reshapedDf)-time,:])
  logReg = LogisticRegressionCV(cv=8, random_state=0).fit(normalizedDf,key)
  prediction = logReg.predict(reshapedDf[:1,:])
  score = str(logReg.score(normalizedDf,key))
  return score + " " + str(prediction[0])

def create_key_for_logistic_regression(n, df):
  closeOfDay = [0] * (len(df[0]) - n)
  for i in range(0,len(df[0])):
      a = df[0][i]
      if i < len(df[0]) - n:
        holder = df[0][i + n]
        closeOfDay[i] = 1 if (a - holder) > 0 else 0
  return closeOfDay

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
  

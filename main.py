import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'] )
def logistic_regression():
    if request.method == 'GET':
      return "Hello"
    if request.method == 'POST':
      body = request.get_json(force=True)
      return create_logistic_regression(body)	

def create_logistic_regression(body):
  sentiment = pd.get_dummies(pd.DataFrame(body['sentiment']).astype('category'), prefix='Test')
  sentiment = check_sentiment_data(sentiment)
  df = np.array([body['close'], body['volume'], body['positive'], body['negative']] + sentiment.T.values.tolist())
  score = {"next": logistic_regression(1,df), "week": logistic_regression(10,df), "month": logistic_regression(20,df), "ticker": body['ticker']}
  return score

def check_sentiment_data(df):
  if(len(df.columns) < 4):
    df["add"] = [0] * len(df)
    if(len(df.columns)<4):
      df["plus"] = [0] * len(df)
  return df

def logistic_regression(time, df):
  key = create_key_for_logistic_regression(time, df)
  processedDf = preprocessing.StandardScaler().fit_transform(df[:len(df)-time,:8])
  logReg = linear_model.LogisticRegressionCV(cv=8, random_state=0).fit(processedDf,key)
  prediction = logReg.predict(df[:1,:8])
  score = str(logReg.score(processedDf,key))
  return score + " " + str(prediction[0])

def create_key_for_logistic_regression(n, df):
  closeOfDay = np.zeros(len(df[0])-n)
  for i, a in enumerate(df[0]):
      if i < len(df[0]) - n:
        holder = df[0][i + n]
        closeOfDay[i] = 1 if (a - holder) > 0 else 0
  return closeOfDay

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
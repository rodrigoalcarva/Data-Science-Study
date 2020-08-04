import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import seaborn as sb
import plotly
import plotly.graph_objs as go
import json
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LinearRegression

 
df = pd.read_csv('PetFinder_dataset.csv')
df = df.drop(['Description','RescuerID','PetID'], axis=1)

ax = sb.heatmap(df.isnull())

def build_graph(aa):
    img = io.BytesIO()
    aa.figure.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

def create_plot1():
    N = 40
    x = [0,1,2,3,4]
    y = [410,3090,4037,3259,4197]
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_plot2():
    N1 = 40
    x1 = [1,2]
    y1 = [8132,6861]
    df1 = pd.DataFrame({'x': x1, 'y': y1}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df1['x'], # assign x as the dataframe column 'x'
            y=df1['y']
        )
    ]

    graphJSON1 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON1



def create_plot3():

    x2 = df['Breed1'].value_counts().keys().tolist()
    y2 = df['Breed1'].value_counts().tolist()   
    df2 = pd.DataFrame({'x': x2, 'y': y2}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df2['x'], # assign x as the dataframe column 'x'
            y=df2['y']
        )
    ]
    graphJSON2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON2
    

def create_plot4():

    x3 = df['Gender'].value_counts().keys().tolist()
    y3 = df['Gender'].value_counts().tolist()   
    df3 = pd.DataFrame({'x': x3, 'y': y3}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df3['x'], # assign x as the dataframe column 'x'
            y=df3['y']
        )
    ]
    graphJSON2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON2

def create_plot5():

    x4 = df['Health'].value_counts().keys().tolist()
    y4 = df['Health'].value_counts().tolist()   
    df4 = pd.DataFrame({'x': x4, 'y': y4}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df4['x'], # assign x as the dataframe column 'x'
            y=df4['y']
        )
    ]
    graphJSON2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON2

def create_plot6():

    x5 = df['Fee'].value_counts().keys().tolist()
    y5 = df['Fee'].value_counts().tolist()   
    df5 = pd.DataFrame({'x': x5, 'y': y5}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df5['x'], # assign x as the dataframe column 'x'
            y=df5['y']
        )
    ]
    graphJSON2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON2

def create_plot7():

    x6 = df['State'].value_counts().keys().tolist()
    y6 = df['State'].value_counts().tolist()   
    df6 = pd.DataFrame({'x': x6, 'y': y6}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df6['x'], # assign x as the dataframe column 'x'
            y=df6['y']
        )
    ]
    graphJSON2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON2
    
df = df.drop(['Name','Color1','Color2','Color3','VideoAmt','PhotoAmt'], axis=1)
def logisticRegression():
    X_train, X_test, y_train, y_test = train_test_split(df, 
                                                    df['AdoptionSpeed'], test_size=0.30, 
                                                    random_state=101)
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions = logmodel.predict(X_test)

    return (classification_report(y_test,predictions))

def linearRegressionn():
    x=df.iloc[:,12:13].values
    y=df.iloc[:,14:15].values
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)
    y_pred=regressor.predict(X_test)

    a = plt.plot(X_test,y_pred)   
    a = plt.scatter(X_test,Y_test,c='red')
    a = plt.xlabel('Fee')
    a = plt.ylabel('AdoptionSpeed') 
    img = io.BytesIO()
    a.figure.savefig(img, format='png')
    img.seek(0)
    graph_url4 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url4)


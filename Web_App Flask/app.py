from flask import Flask, render_template, request, send_file
from graph import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly
import plotly.graph_objs as go
import json

app = Flask(__name__)


df = pd.read_csv('PetFinder_dataset.csv')
df = df.drop(['Description','RescuerID','PetID'], axis=1)
df = df.head()

ax = sb.heatmap(df.isnull())

TypeCount = df['Type'].value_counts()
TypePerc = df['Type'].value_counts()/ sum(TypeCount) * 100 

AgeCount = df['Age'].value_counts()
AgePerc = df['Age'].value_counts()/ sum(AgeCount) * 100

Breed1Count = df['Breed1'].value_counts()
Breed1Perc = df['Breed1'].value_counts()/ sum(Breed1Count) * 100

Breed2Count = df['Breed2'].value_counts()
Breed2Perc = df['Breed2'].value_counts()/ sum(Breed2Count) * 100

GenderCount = df['Gender'].value_counts()
GenderPerc = df['Gender'].value_counts()/ sum(GenderCount) * 100

MaturityCount = df['MaturitySize'].value_counts()
MaturityPerc = df['MaturitySize'].value_counts()/ sum(MaturityCount) * 100

FurLengthCount = df['FurLength'].value_counts()
FurLengthPerc = df['FurLength'].value_counts()/ sum(FurLengthCount) * 100

VaccinatedCount = df['Vaccinated'].value_counts()
VaccinatedPerc = df['Vaccinated'].value_counts()/ sum(VaccinatedCount) * 100

DewormedCount = df['Dewormed'].value_counts()
DewormedPerc = df['Dewormed'].value_counts()/ sum(DewormedCount) * 100

SterilizedCount = df['Sterilized'].value_counts()
SterilizedPerc = df['Sterilized'].value_counts()/ sum(SterilizedCount) * 100

HealthCount = df['Health'].value_counts()
HealthPerc = df['Health'].value_counts()/ sum(HealthCount) * 100

StateCount = df['State'].value_counts()
StatePerc = df['State'].value_counts()/ sum(StateCount) * 100

AdoptionSpeedCount = df['AdoptionSpeed'].value_counts()
AdoptionSpeedPerc = df['AdoptionSpeed'].value_counts()/ sum(AdoptionSpeedCount) * 100 

textAge = 'A variância de Age é : ' + str(df['Age'].var())
textBreed1 = 'A variância de Breed1 é : ' + str(df['Breed1'].var())
textBreed2 = 'A variância de Breed2 é : ' + str(df['Breed2'].var())
textGender = 'A variância de Gender é : ' + str(df['Gender'].var())
textColor1 = 'A variância de Color1 é : ' + str(df['Color1'].var())
textColor2 = 'A variância de Color2 é : ' + str(df['Color2'].var())
textColor3 = 'A variância de Color3 é : ' + str(df['Color3'].var())
textMaturity = 'A variância de MaturitySize é : ' + str(df['MaturitySize'].var())
textFurLength = 'A variância de FurLength é : ' + str(df['FurLength'].var())
textVaccinated = 'A variância de Vaccinated é : ' + str(df['Vaccinated'].var())
textDewormed = 'A variância de Dewormed é : ' + str(df['Dewormed'].var())
textSterilized = 'A variância de Sterilized é : ' + str(df['Sterilized'].var())
textHealth = 'A variância de Health é : ' + str(df['Health'].var())
textQuantity = 'A variância de Quantity é : ' + str(df['Quantity'].var())
textFee = 'A variância de Fee é : ' + str(df['Fee'].var())
textState = 'A variância de State é : ' + str(df['State'].var())
textVideoAmt = 'A variância de VideoAmt é : ' + str(df['VideoAmt'].var())
textPhotoAmt = 'A variância de PhotoAmt é : ' + str(df['PhotoAmt'].var())
textAdoptionSpeed = 'A variância de AdoptionSpeed é : ' + str(df['AdoptionSpeed'].var())

pearsoncorr = df.corr(method='pearson')
plt.switch_backend('agg')
plt.figure(figsize = (20,10))
axs = sb.heatmap(pearsoncorr,xticklabels=pearsoncorr.columns, yticklabels=pearsoncorr.columns,cmap='RdBu_r',annot=True, linewidth=0.05)
bottom, top = axs.get_ylim()
axs.set_ylim(bottom + 0.5, top - 0.5)

plt.figure(figsize=(10, 6))
axsss = sb.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=df)
plt.title('AdoptionSpeed by Type and age')



@app.route("/", methods = ["GET", "POST"])
def prehome():
    return render_template("prehome.html")

@app.route("/home", methods = ["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/trabalho", methods = ["GET", "POST"])
def trabalho():
    return render_template("trabalho.html")

@app.route("/tema", methods = ["GET", "POST"])
def tema():
    return render_template("tema.html")

@app.route("/grupo", methods = ["GET", "POST"])
def grupo():
    return render_template("grupo.html")

@app.route("/trabalho/datapreparation", methods = ["GET", "POST"])
def datapreparation():
    graph1_url = build_graph(ax)
    graph2_url = build_graph(axs)
    return render_template("datapreparation.html", tables=[df.to_html(classes='data')], titles=df.columns.values, graph1=graph1_url, graph2=graph2_url,
    TypeCount = TypeCount, TypePerc =TypePerc, AgeCount = AgeCount, AgePerc = AgePerc, Breed1Count = Breed1Count,  Breed1Perc = Breed1Perc,
    Breed2Count = Breed2Count, Breed2Perc = Breed2Perc, GenderCount = GenderCount, GenderPerc = GenderPerc, MaturityCount = MaturityCount, 
    MaturityPerc = MaturityPerc, FurLengthCount = FurLengthCount, FurLengthPerc = FurLengthPerc,
    VaccinatedCount = VaccinatedCount, VaccinatedPerc = VaccinatedPerc, DewormedCount = DewormedCount, DewormedPerc = DewormedPerc,
    SterilizedCount = SterilizedCount, SterilizedPerc = SterilizedPerc, HealthCount = HealthCount, HealthPerc = HealthPerc, 
    StateCount = StateCount, StatePerc = StatePerc, AdoptionSpeedCount = AdoptionSpeedCount, AdoptionSpeedPerc = AdoptionSpeedPerc,
    textAge = textAge, textBreed1 = textBreed1, textBreed2 = textBreed2, textGender = textGender, textColor1 = textColor1, textColor2 = textColor2,
    textColor3 = textColor3, textMaturity = textMaturity, textFurLength = textFurLength, textVaccinated = textVaccinated, textDewormed = textDewormed,
    textSterilized = textSterilized, textHealth = textHealth, textQuantity = textQuantity, textFee = textFee, textState = textState, textVideoAmt = textVideoAmt,
    textPhotoAmt = textPhotoAmt, textAdoptionSpeed = textAdoptionSpeed)


@app.route("/trabalho/datagraphics", methods = ["GET", "POST"])
def datagraphics():
    bar = create_plot1()
    bar1 = create_plot2()
    bar2 = create_plot3()
    bar3 = create_plot4()
    bar4 = create_plot5()
    bar5 = create_plot6()
    bar6 = create_plot7()
    graph3_url = build_graph(axsss)
    return render_template("datagraphics.html", plot=bar, plot1 = bar1, plot2 = bar2, plot3 = bar3, plot4 = bar4, plot5 = bar5, plot6 = bar6, graph3 = graph3_url)

@app.route("/trabalho/dataexploration", methods = ["GET", "POST"])
def dataexploration():
    log = logisticRegression()
    graph4_url = linearRegressionn()
    return render_template("dataexploration.html", log = log, graph4 = graph4_url)

if __name__ == "__main__":
    app.run()

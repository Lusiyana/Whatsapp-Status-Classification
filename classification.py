import pandas as pd
import math
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Read .csv data
dataframes = {"data": pd.read_csv("dataWA.csv")}

#Preprocessing
def cleansing(text):
    regexURL = re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})', '', text)
    regexUsername = re.sub(r'\S*@\S*\s?', '', regexURL)
    regexContainNumber = re.sub(r'\w*[0-9]\w*', '', regexUsername)
    regexNotAlphabet = re.sub('[^A-Za-z]+', ' ', regexContainNumber)
    return regexNotAlphabet

def casefolding(text):
    return text.lower()

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
def removeStopwords(x):
    filtered_words = [word for word in x.split() if word not in stopwords]
    return filtered_words

def stemming(x):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    documents = [stemmer.stem(word) for word in x]
    return documents

for df in dataframes.values():
    df["Teks"] = df["Teks"].map(cleansing)
    
for df in dataframes.values():
    df["Teks"] = df["Teks"].map(casefolding)
    
for df in dataframes.values():
    df["Teks"] = df["Teks"].map(removeStopwords)
    
for df in dataframes.values():
    df["Teks"] = df["Teks"].map(stemming)

x = dataframes["data"]["Teks"]
y = dataframes["data"]["label"]


############To test the normalized data#########
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=5)

############To test using k-fold cross validation###########
'''
k=2
kf=KFold(n_splits=k, shuffle=True)
print(kf)  #buat tau Kfold dan parameter defaultnya
i=1        #ini gapenting, cuma buat nandain fold nya.

for train_index, test_index in kf.split(x):
    print("Fold ", i)
    print("TRAIN :", train_index, "TEST :", test_index)
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    i+=1

print("shape x_train :", x_train.shape)
print(x_test)
print("shape x_test :", x_test.shape)
'''

datalatih = x_train.reset_index(drop = True)
datauji = x_test.reset_index(drop = True)

#Get unique terms
def term(teks):
    unik = []
    for i in teks:
        for j in i:
            if j not in unik:
                unik.append(j)
    return unik

termlatih = term(datalatih)
termuji = term(datauji)

#Calculate the TF value of data training
def hitungtflatih(teks):
    tf = []
    #for i in term(teks):
    for i in termlatih:
        temp_tf = []
        for j in teks:
            jml=0
            for k in j:
                if(k==i):
                    jml+=1
            if jml!= 0:
                jml = 1+math.log10(jml)
            temp_tf.append(jml)
        tf.append(temp_tf)
    return tf

#Calculate the TF value of data testing
def hitungtfuji(latih, uji):
    arr = []
    #for i in range(len(term(latih))):
    for i in range(len(termlatih)):
        a = []
        for j in range(len(uji)):
            a.append(0)
            for k in range(len(uji[j])):
                if (termlatih[i] == uji[j][k]):
                    a[j] += 1
            if a[j]!=0:
                a[j] = 1+math.log10(a[j])
        arr.append(a)
    return arr

tflatih = hitungtflatih(datalatih)
tfuji = hitungtfuji(datalatih,datauji)
tlatih = np.transpose(hitungtflatih(datalatih))
tuji = np.transpose(hitungtfuji(datalatih,datauji))

#Calculate the DF value
def hitungdf(teks):
    df = []
    for i in termlatih:
    #for i in term(teks):
        jml_dok = 0
        for j in teks:
            jml=0
            for k in j:
                if(k==i):
                    jml+=1
            if jml!= 0:
                jml_dok += 1
                jml = 1+math.log10(jml)
        df.append(jml_dok)
    return df

dflatih = hitungdf(datalatih)

#Calculate the IDF value
def hitungidf(teks):
    idf = []
    for i in dflatih:
    #for i in hitungdf(teks):
        idf.append(math.log10(len(teks)/i))
    return idf

idflatih = hitungidf(datalatih)

#Calculate the TF-IDF value
def hitungtfidf(teks):
    if (teks is datalatih):
        tf = tflatih
    else:
        tf = tfuji
    idf = idflatih
    tfidf = []
    for i in range(len(tf)):
        temp_tfidf = []
        for j in range(len(tf[i])):
            temp_tfidf.append(tf[i][j]*idf[i])
        tfidf.append(temp_tfidf)
    return tfidf

tfidflatih = np.transpose(hitungtfidf(datalatih))
tfidfuji = np.transpose(hitungtfidf(datauji))

#Calculate the normalization from TF-IDF
def normalisasi(teks):
    tfidf = hitungtfidf(teks)
    norm = []
    
    temp_hasilakar=[] 
    n=0
    while(n<len(teks)):
        tambahwtd2=0
        for i in range(len(tfidf)):
            tambahwtd2 += (tfidf[i][n])**2
        akarwtd2 = math.sqrt(tambahwtd2)
        temp_hasilakar.append(akarwtd2)
        n += 1
    
    for i in range(len(tfidf)):
        temp_norm = []
        for j in range(len(tfidf[i])):
            temp_norm.append(tfidf[i][j]/temp_hasilakar[j])
        norm.append(temp_norm)
    return norm

normlatih = np.transpose(normalisasi(datalatih))
normuji = np.transpose(normalisasi(datauji))

############To test using k-fold cross validation#########
'''
clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(normlatih, y_train)
#print(clf.predict(normuji))

score = []
n_k=[]

n_k.append(k)
s = cross_val_score(clf, normlatih, y_train, cv=k, scoring='accuracy') 
score.append(s.mean()*100)
print(score, n_k)
k+=1
print(end - start)
while (k<11):
    print(k)
    n_k.append(k)
    s = cross_val_score(clf, normlatih, y_train, cv=k, scoring='accuracy') 
    score.append(s.mean()*100)
    k+=2
print(score, n_k)

plt.plot(n_k, score, color='#226089', linestyle='-', linewidth = 3, marker='o', markerfacecolor='#4592af', markersize=12) 
  
# setting x and y axis range 
plt.ylim(40,90) 
plt.xlim(0,12) 
  
# naming the x axis 
plt.xlabel('Nilai K') 
# naming the y axis 
plt.ylabel('Akurasi (%)') 
  
# giving a title to my graph 
plt.title('Akurasi K-Fold') 
  
# function to show the plot 
plt.show() 
'''


############To test the normalized data#########
clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(tlatih, y_train)
hasiluji = clf.predict(tuji)
print(hasiluji)
labelujisebenarnya = y_test.reset_index(drop = True)

#Calculate the accuration
def ujiakurasi(hasiluji, labelujisebenarnya):
    akurasi=0
    for i in range(len(labelujisebenarnya)):
        if(hasiluji[i] == labelujisebenarnya[i]):
            akurasi +=100
    return akurasi/(len(labelujisebenarnya))

hasilakurasi = ujiakurasi(hasiluji,labelujisebenarnya)
print('Hasil Akurasi: ',hasilakurasi)

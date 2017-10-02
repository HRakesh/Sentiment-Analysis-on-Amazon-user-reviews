import csv
import re
import string
import pandas as pd
from sklearn.svm import SVC
from collections import Counter
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.ensemble import GradientBoostingClassifier as gbc

stopWords = []
dataSet = []
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
    r'<[^>]+>',  # HTML tags
    r"(?:[a-z][a-z\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

def initializeSystem():
    stop = stopwords.words('english') + list(string.punctuation) + ['rt', 'via', 'i\'m', 'us', 'it']
    for x in stop:
        stopWords.append(stemmer.stem(lemmatiser.lemmatize(x, pos="v")))

def preprocess(s, lowercase=True):
    tokens = tokens_re.findall(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else stemmer.stem(lemmatiser.lemmatize(token.lower(), pos="v")) for
                  token in tokens]
    return tokens

def processString(string):
    terms_stop = [term for term in preprocess(string) if
                  term not in stopWords and len(str(term)) > 1 and not term.isnumeric()]
    return terms_stop

def loadFile(filePath):
    fileRead = open(filePath, "r")
    reader = csv.reader(fileRead, dialect='excel')
    for row in reader:
        temp = (row[1], row[-1])
        dataSet.append(temp)
    return dataSet

def prepareSparseMatrix(convertedReviews, decisionAttributes):
    sparseMatrix = []
    for cr in convertedReviews:
        newCr = [0] * len(decisionAttributes)
        for word in cr:
            if word in decisionAttributes:
                index = decisionAttributes.index(word)
                newCr[index] += 1
            else:
                pass
        sparseMatrix.append(newCr)
    return sparseMatrix

def convertReviews(reviews):
    convertedReviews = []
    for a in reviews:
        convertedReviews.append(processString(str(a).lower()))
    return convertedReviews

def getDecisionAttributes(convertedReviews):
    toCount = []
    decisionAttributes = []
    for a in convertedReviews:
        toCount.append(" ".join(a))
    str1 = ""
    for a in toCount:
        str1 += "".join(a)
    x = Counter(str1.split(" "))
    for (k, v) in x.most_common(min(500, len(x))):
        decisionAttributes.append(k)
    return decisionAttributes

def model_data(training_data):
    dtc = DecisionTreeClassifier(random_state=9, min_samples_split=5)
    dtc.fit(training_data['data'], training_data['result'])

    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    nn.fit(training_data['data'], training_data['result'])

    svc = SVC(C=100, kernel="linear")
    svc.fit(training_data['data'], training_data['result'])

    rfc = RFC(n_estimators=10, criterion='entropy', max_depth=10, min_samples_split=5, bootstrap='true', random_state=None)
    rfc.fit(training_data['data'], training_data['result'])


    knc_map = knc(n_neighbors=15, weights='distance')
    knc_map.fit(training_data['data'], training_data['result'])

    gbc_map = gbc(n_estimators=150, verbose=0)
    gbc_map.fit(training_data['data'], training_data['result'])

    return {
        'Decision Tree Classifier': dtc,
        'Neural Networks': nn,
        'Support Vector Machines': svc,
        'Random Forest Classification': rfc,
        'k Nearest Neighbours': knc_map,
        'Gradient Boosting Classifier': gbc_map
    }

def test_models(test_data, models):
    print("Prediction rating:\n")
    for model in models:
        prediction = models[model].score(test_data['data'], test_data['result'])*100.00
        print(str(model) + ": " + "%.2f" % prediction + "%")


initializeSystem()

training_data = loadFile("../data/training_data_small.csv")
trainDataFeaturesReviews = pd.DataFrame(training_data, columns=["review", "rating"])
targetRating = (trainDataFeaturesReviews['rating'])
targetReview = trainDataFeaturesReviews['review']
trainReviews = convertReviews(targetReview)
decisionAttributes = getDecisionAttributes(trainReviews)
trainSparseMatrix = prepareSparseMatrix(trainReviews, decisionAttributes)
dataFeatures = pd.DataFrame(trainSparseMatrix, columns=decisionAttributes)
training_data = {
    'data': dataFeatures,
    'result': targetRating
}

test_data = loadFile("../data/test_data_small.csv")
testDataFeaturesReviews = pd.DataFrame(test_data, columns=["review", "rating"])
testReview = testDataFeaturesReviews['review']
testRating = testDataFeaturesReviews['rating']
testSparseMatrix = prepareSparseMatrix(convertReviews(testReview), decisionAttributes)
testDataFeatures = pd.DataFrame(testSparseMatrix, columns=decisionAttributes)
test_data = {
    'data': testDataFeatures,
    'result': testRating
}

models = model_data(training_data)
test_models(test_data, models)



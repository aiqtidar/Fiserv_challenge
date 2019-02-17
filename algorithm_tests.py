# Load libraries
import account
import time
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



#Defines a class for checking algorithm
class algorithm_tests:

    def import_discrete_dataset(self):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.dataset = pandas.read_csv(url, names=names)        


    def import_continous_dataset(self):
        names = ['month', 'clothing', 'food', 'trans', 'misc']
        self.dataset = pandas.read_csv('spending.csv', delimiter = ',')
    
    def evaluate_dataset(self):
        # Define validation dataset
        array = self.dataset.values
        X = array[:,1:4]
        Y = array[:,4]
        validation_size = 0 #Take entire dataset for now
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

        seed = 7
        scoring = 'accuracy'

        #--------------------------------------------------------------------------------------------
        '''
        Run Six different machine learning algorithms one after another and evaluate their performance:
        Logistic Regression (LR)
        Linear Discriminant Analysis (LDA)
        K-Nearest Neighbors (KNN)
        Classification and Regression Trees (CART)
        Gaussian Naive Bayes (NB)
        Support Vector Machines (SVM)
        '''
        
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
                algorithm_tests.start_time(self)
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                print(msg)
                algorithm_tests.end_time(self)
                algorithm_tests.print_runtime(self)
        
    def start_time(self):
            self.start = time. time()

    def end_time(self):
            self.end = time. time()

    def print_runtime(self):
            print ("\nRuntime: ", self.end - self.start)


#Calls functions to output performances
def main():
    naima = algorithm_tests()

    print("Result on discrete data set: \n")
    #Check on discrete data set
    naima.import_discrete_dataset()
    naima.evaluate_dataset()

    print("\n")
    
    print("\nResult on continous data set: \n")
    #Check on continous data set
    naima.import_continous_dataset()
    naima.evaluate_dataset()  

if __name__=='__main__':
    main()

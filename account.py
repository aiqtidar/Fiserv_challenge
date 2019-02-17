#This is the account class which defines an account and contains the full range of functions for predicting the user's spending

import pandas
import sklearn
from sklearn.linear_model import LinearRegression
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


class account:
    monthly_interest = 0.02
    higher_interest = 0.04
    lower_interest = 0.01
    def __init__(self, b = 0, location = 0,recordID=""):
        self.recordID = recordID
        self.balance = b
        self.coefficients = []
        if location != 0:
            account.import_dataset(self, location)
        else:
            dataset = []
        account.average_year(self)
        self.spent = 0;
        self.exp_spending = 0
        self.interest = 0
        self.exp_spendingupper = 10
        self.exp_spendinglower = 10

    def get_balance(self):
        print (self.balance)

    def add_balance(self,amount = 0):
        self.balance = self.balance + amount
        print ("New balance:",self.balance)
        return
    
    #Each account has it's own dataset
    def import_dataset(self,location):
        self.dataset = pandas.read_csv(location, delimiter = ',')
        


    def weighted_mean(self,month):
        array = self.dataset.values
        month_array = array[:,0]
        clothing_array = array[:,1]
        food_array = array[:,2]
        trans_array = array[:,3]
        misc_array = array[:,4]
        mean = 0

        total_cloth = 0
        total_food = 0
        total_trans = 0
        total_misc = 0

        for i in range(self.dataset.shape[0]):
            if (month_array[i] == month):
                total_cloth = total_cloth + clothing_array[i]
                total_food = total_food + food_array[i]
                total_trans = total_trans + trans_array[i]
                total_misc = total_misc + misc_array[i]
        mean = (total_cloth + total_food + total_trans + total_misc)/float(self.dataset.shape[0])
        return (mean)

    def average_year(self):
        array = self.dataset.values
        total_array = array[:,5]
        total = 0
        for i in range(self.dataset.shape[0]):
            total = total + total_array[i]
        self.year_mean = (total)/float((self.dataset.shape[0]))
        return (self.year_mean)
    
    #This is the simplest algorithm implementing Linear Regression. The key feature for performance would be selecting the variables we expect dependence on.
    #I have very little knowledge on finance, and I certainly failed to account for implicit interdependece between the variables I have chosen. Results will
    #certainly improve if the chosen variables are changed.
    def evaluate_spending(self):
        array = self.dataset.values
        total_array = array[:,5]
        month_array = array[:,0]
        
        TRAIN_SET_COUNT = self.dataset.shape[0]
        TRAIN_INPUT = list()
        TRAIN_OUTPUT = list()
        for i in range(TRAIN_SET_COUNT):
            TRAIN_INPUT.append([account.weighted_mean(self,month_array[i]), self.year_mean, month_array[i]])
            TRAIN_OUTPUT.append(total_array[i])

        predictor = LinearRegression(n_jobs=-1)
        predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

        self.coefficients = predictor.coef_
        return
    
    def predict_spending(self,month):
        self.exp_spending = (self.coefficients[0]*account.weighted_mean(self,month)+self.coefficients[1]*self.year_mean+self.coefficients[2]*month)
        self.exp_spendingupper = account.calculate_spendingupper(self)
        self.exp_spendingupper = account.calculate_spendinglower(self)
        return self.exp_spending
    
    def make_payment(self,amount,month_end = False):
        self.spent = self.spent + amount
        self.balance = self.balance - amount
        if (month_end == True):
            if (self.spent > float(self.exp_spendingupper)) and (self.upperbound == True):
                account.calculate_interest(self, higher_interest, self.spent - self.exp_spending)
                print ("Spending too much. Further option to disable card")
            elif self.spent < float(self.exp_spendinglower) and (self.lowerbound == True):
                account.calculate_interest(self, lower_interest, self.exp_spending-self.spent)
                print ("Spending too little")
            
    def set_upperbound(self,boolean = False):
        self.upperbound = boolean

    def set_lowerbound(self,boolean = False):
        self.lowerbound = boolean

    def calculate_interest(self,rate,amount):
        self.interest = rate*amount

    def calculate_spendingupper(self):
        #The higher the value, the better. This is just an arbitrary high value. I'd set a value and look at customer feedback to select a value.
        self.exp_spendingupper = self.exp_spending + 100

    def calculate_spendinglower(self):
        #The lower the value, the better. This is just an arbitrary  low value. I'd set a value and look at customer feedback to select a value.
        self.exp_spendinglower = self.exp_spending + 10
        
    def return_recordID(self):
        return self.recordID

    

#Calls functions to check for errors.
def main():
    naima = account(1000,'spending.csv',10)
    naima.get_balance()
    naima.add_balance(10)
    naima.evaluate_spending()
    print(naima.coefficients)
    for i in range (1,13):
        print(i," ",naima.predict_spending(i))
    print(naima.return_recordID())

    naima.make_payment(200,False)
    naima.get_balance()
   

if __name__=='__main__':
    main()

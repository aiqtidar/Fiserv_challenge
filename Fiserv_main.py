import account

global n
global records


#This is the terminal for creating records
def main():
    global n
    n = 0
    records = []
    while True:
        print("What would you like to do?")
        print("1. Create a record")
        print("2. Get balance for a record ")
        print("3. Add balance for a record ")
        print("4. Make a payment")
        print("5. Evaluate spending patterns")
        print("6. Predict spending patterns for a month")
        
        x = int(input("Please print a choice: "))
        if x == 1:
            recordID = input("Please enter RecordID: ")
            location = input("Input dataset name: ")
            balanc = input("Please enter initial balance: ")
            records.append(account(balanc,location,recordID))
        elif x == 2:
            recID = input("Please enter RecordID: ")
            for i in records:
                if i.recordID == recID:
                    i.get_balance()
        elif x == 3:
            recID = input("Please enter RecordID: ")
            add = input("Please enter amount: ")
            for i in records:
                if i.recordID == recID:
                    i.add_balance(add)
        elif x == 4:
            recID = input("Please enter RecordID: ")
            sub = input("Please enter amount: ")
            end = bool(input("Is it end of month? True/False"))
            for i in records:
                if i.recordID == recID:
                    i.make_payment(sub,end)
        elif x == 5:
            recID = input("Please enter RecordID: ")
            for i in records:
                if i.recordID == recID:
                    i.evaluate_spending()
        elif x == 6:
            recID = input("Please enter RecordID: ")
            month = int(input("Please enter month (1 to 12): "))
            for i in records:
                if i.recordID == recID:
                    i.predict_payment(month)
        else:
            print("Invalid Input")

if __name__=='__main__':
    main()


            
            
                    
        

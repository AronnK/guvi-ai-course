import numpy as np
from prettytable import PrettyTable
friends = ["A", "B", "C", "D"]

expense_matrix = np.zeros((len(friends), len(friends)))

def add_expense(payer, beneficiaries, amount):

    print(f"Adding expense: {payer} paid ₹{amount} for {', '.join(beneficiaries)}")
    payer_idx = friends.index(payer)
    share_per_person = amount / len(beneficiaries)
    for beneficiary in beneficiaries:
        beneficiary_idx = friends.index(beneficiary)
        expense_matrix[payer_idx][beneficiary_idx] += share_per_person

def calculate_settlements():

    total_paid = np.sum(expense_matrix, axis=1)
    total_owed = np.sum(expense_matrix, axis=0)
    net_balance = total_paid - total_owed
    return net_balance

def display_settlements():

    settlements = calculate_settlements()
    table = PrettyTable()
    table.field_names = ["Friend", "Settlement Status"]
    
    for i, friend in enumerate(friends):
        if settlements[i] > 0:
            table.add_row([friend, f"Should Receive ₹{settlements[i]:.2f}"])
        elif settlements[i] < 0:
            table.add_row([friend, f"Owes ₹{-settlements[i]:.2f}"])
        else:
            table.add_row([friend, "Is Settled"])
            
    print("\nFinal Settlements:")
    print(table)

def suggest_payments():

    settlements = calculate_settlements()
    creditors = [[friends[i], amt] for i, amt in enumerate(settlements) if amt > 0]
    debtors = [[friends[i], -amt] for i, amt in enumerate(settlements) if amt < 0]
    
    transactions = []

    while debtors and creditors:
        debtor, debt_amount = debtors.pop(0)
        creditor, credit_amount = creditors.pop(0)

        payment = min(debt_amount, credit_amount)
        transactions.append((debtor, creditor, payment))

        debt_amount -= payment
        credit_amount -= payment
        if debt_amount > 0:
            debtors.insert(0, (debtor, debt_amount))
        if credit_amount > 0:
            creditors.insert(0, (creditor, credit_amount))

    print("\nSuggested Transactions:")
    if transactions:
        for debtor, creditor, amount in transactions:
            print(f"- {debtor} should pay ₹{amount:.2f} to {creditor}")
    else:
        print("No transactions needed. Everyone is settled.")


add_expense("A", ["A", "B", "C"], 2534)
add_expense("B", ["B", "C"], 987)
add_expense("C", ["A", "B", "C", "D"], 490)

display_settlements()
suggest_payments()
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Reading data from dataset
data = pd.read_csv("216771875-216328387-215122856-T1DiscTF.csv")
# Extracting the most frequest itemsets via Mlxtend.
# Calling apriori function on dataset with minimum support = 0.4
frequent_itemsets = apriori(data, min_support=0.1, use_colnames=True)
# Adding length columns to present the length of the itemset
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
# Print out the 10 most frequent itemset that has length at least 2
print(frequent_itemsets[(frequent_itemsets["length"] >= 2)].head(10).sort_values("support", ascending=False))

lista = frequent_itemsets[(frequent_itemsets["length"] >= 2)].head(10).sort_values("support",
                                                                                   ascending=False).itemsets.to_list()
print(lista)

# Calling association rules function on frequent_itemset with the metric of confidence = 0.1 for least confidence
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
# Adding column for the length of the LHS
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
# Adding column for the length of the RHS
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
# rules.sort_values("confidence", ascending=False)
# Printing out sorted rules
# print(rules.to_string())
print(rules[(rules['antecedents'] == {'paid'}) & ((
                                                          rules['consequents'] == {'G[5,10)'}) | (
                                                          rules['consequents'] == {'G[10,15)'}) | (
                                                          rules['consequents'] == {'G[15,20)'}))].to_string())

print(rules[(rules['antecedents'] == {'famsup'}) & ((
                                                          rules['consequents'] == {'G[5,10)'}) | (
                                                          rules['consequents'] == {'G[10,15)'}) | (
                                                          rules['consequents'] == {'G[15,20)'}))].to_string())

print(rules[(rules['antecedents'] == {'schoolsup'}) & ((
                                                          rules['consequents'] == {'G[5,10)'}) | (
                                                          rules['consequents'] == {'G[10,15)'}) | (
                                                          rules['consequents'] == {'G[15,20)'}))].to_string())

print(rules[(rules['antecedents'] == {'famsup','schoolsup'}) & ((
                                                          rules['consequents'] == {'G[5,10)'}) | (
                                                          rules['consequents'] == {'G[10,15)'}) | (
                                                          rules['consequents'] == {'G[15,20)'}))].to_string())

# Sorting generated rules by the value of confidence in order
# rules[rules['antecedents'] == {'Eggs', 'Kidney Beans'}]

import higgs
import numpy as np
from higgs import*

# Read the result.csv file
result_df = pd.read_csv('result.csv')

# Count occurrences of 'b' and 's' in the Predicted_Label column
count_b = result_df['Predicted_Label'].value_counts()['b']
count_s = result_df['Predicted_Label'].value_counts()['s']

print("Number of 'b':", count_b)
print("Number of 's':", count_s)

def ams(b,s):
    return np.sqrt(2*((s+b+10)*np.log(1+s/(b+10))-s ) )

x=ams(531598,18402)
print(x)
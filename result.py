import higgs
from higgs import*

# Preprocess test data
test_data.replace(-999.0, np.nan, inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Separate features and IDs
test_ids = test_data['EventId']
X_test_data = test_data.drop(['EventId'], axis=1)

# Standardize the features of the test data
X_test_data = scaler.transform(X_test_data)

# Make predictions on the test data
test_predictions = model.predict(X_test_data)

# Convert predictions back to original labels
predicted_labels = label_encoder.inverse_transform(test_predictions)

# Create a DataFrame for results
results_df = pd.DataFrame({'EventId': test_ids, 'Predicted_Label': predicted_labels})

# Save results to CSV
results_df.to_csv('result.csv', index=False)

# Read the result.csv file
result_df = pd.read_csv('result.csv')

# Count occurrences of 'b' and 's' in the Predicted_Label column
count_b = result_df['Predicted_Label'].value_counts()['b']
count_s = result_df['Predicted_Label'].value_counts()['s']

print("Number of 'b':", count_b)
print("Number of 's':", count_s)

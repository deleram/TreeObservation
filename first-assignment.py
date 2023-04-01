# importing needed libraries
import pandas as pd
from sklearn.model_selection import train_test_split
# First step which is to load and dividing data to X and Y
dfPath = r'data.csv'
assignmentData = pd.read_csv(dfPath)
print(assignmentData.head())

# Check if any rows have null values
null_rows = assignmentData.isnull().any(axis=1)
print('Number of rows with null values:', sum(null_rows))
#  There were no null values

# Check if there is any duplicate rows
duplicate_rows = assignmentData[assignmentData.duplicated()]
# Print the number of duplicate rows
print("Number of duplicate rows: ", len(duplicate_rows))
# none where found

X = assignmentData.drop(columns=['Cover_Type'])
y = assignmentData['Cover_Type']

print(X.head())
print(y.head())

# Divide data into train and test set. We chose test_size = 0.3 to get a 70-30 division.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,shuffle=True)

#Second step : It's time to preproccess
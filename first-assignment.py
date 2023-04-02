# importing needed libraries
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
print("Number of duplicate rows: ", len(duplicate_rows))
# none where found

X = assignmentData.drop(columns=['Cover_Type'])
y = assignmentData['Cover_Type']

print(X.head())
print(y.head())

# Second step : Divide data into train and test set. We chose test_size = 0.3 to get a 70-30 division.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)

# Third step : It's time to preproccess
# As we checked there is no missing values and no duplicate ones.
# And all of our data is numerical, and there are no categorical columns that need to be converted to numerical values.
# So we should just check if scaling is needed.

# As we can see from the head our columns are not from the same range so we should scale the large ones
# choose desired columns
cols_to_scale = ['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology',
                 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
                 'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
#define our scaler
Scaler = StandardScaler()
# scale our columns
Scaler.fit(X_train[cols_to_scale])
X_train[cols_to_scale] = Scaler.transform(X_train[cols_to_scale])
X_test[cols_to_scale] = Scaler.transform(X_test[cols_to_scale])

print(X_train.head())

# Step 4: 5 fold validation
five_fold = KFold(n_splits=5, shuffle=True, random_state=10)

tree_parameters ={'max_depth': [20, 25],
                    'criterion' : ['entropy', 'log_loss'],
                    'min_samples_split': [5, 7],
                    'min_samples_leaf' : [1, 2]}

# Create a decision tree classifier model
# tree_model = DecisionTreeClassifier()

# # search for the best parameters
# tree_grid = GridSearchCV(tree_model, tree_parameters, cv=five_fold, error_score="raise")

# # Fit the GridSearchCV objects to your data
# tree_grid.fit(X_train, y_train)

# print("Best parameters for Decision Tree: ", tree_grid.best_params_)

# # Printing all othe options:
# results_df = pd.DataFrame(tree_grid.cv_results_)
# results_df = results_df[['params', 'mean_test_score', 'std_test_score']]
# results_df = results_df.sort_values(by=['mean_test_score'], ascending=False)
# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', None,
#                        'display.precision', 3,
#                        'display.max_colwidth', None
#                        ):
#     print(results_df)

Best_tree = {'criterion': 'log_loss', 'min_samples_leaf': 1, 'min_samples_split': 5}


# Same thing for random forest
forest_parameters =  {'n_estimators': [100, 150],
                    'criterion' : ['entropy'],
                    'min_samples_split': [5, 7],
                    'min_samples_leaf' : [1, 2],
                    }

# # Create a random forest classifier model
# forest_model = RandomForestClassifier()
# forest_grid = GridSearchCV(forest_model, forest_parameters, cv=five_fold)
# forest_grid.fit(X_train, y_train)

# # Print the best parameters for each model

# print("Best parameters for Random Forest: ", forest_grid.best_params_)

# results_df = pd.DataFrame(forest_grid.cv_results_)
# results_df = results_df[['params', 'mean_test_score', 'std_test_score']]
# results_df = results_df.sort_values(by=['mean_test_score'], ascending=False)
# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', None,
#                        'display.precision', 3,
#                        'display.max_colwidth', None
#                        ):
#     print(results_df)

Best_forest = {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 150}
# fit the model to our test data to see the results

#  Step 5 & Documentation : prediction

tree_model = DecisionTreeClassifier(criterion = "log_loss" , min_samples_leaf = 1, min_samples_split = 5)
tree_model.fit(X_train,y_train)
y_pred = tree_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("------------------------------------")

forest_model = RandomForestClassifier(criterion = "entropy" , min_samples_leaf = 1, min_samples_split = 5, n_estimators= 150 )
forest_model.fit(X_train,y_train)
y_pred = forest_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


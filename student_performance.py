import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics


# Loading the dataset is in a CSV file named 'dataset.csv'

data = pd.read_csv('student_digital_life.csv')

# Separate features and target variable

X = data.drop('target', axis=1) # Replace 'target' with the actual name of your target variable.
y = data['target'] # Replace 'target' with the actual name of your target variable.












# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier

clf = DecisionTreeClassifier(random_state=42)

# Train the classifier

clf.fit(X_train, y_train)

# Make predictions on the test set

y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')  

# visualizing the decision tree

tree.plot_tree(clf)

# Printing a classification report

print(metrics.classification_report(y_test, y_pred))






import os
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import dtreeviz
import matplotlib.pyplot as plt

# Setting global available fonts

os.environ['FONTCONFIG_PATH'] = '/usr/share/fonts' 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']



# Loading the dataset is in a CSV file named 'dataset.csv'

student_df = pd.read_csv('student_digital_life.csv')
print(student_df.shape)
print(student_df.head())
print(student_df.info())
print(student_df.describe())
print(student_df.columns.tolist())

# Basic Exploration:

print(student_df['final_exam_score'].describe())    
print("\nMental Health:", student_df['mental_health_status'].value_counts())
print("\nsmartphone_usage_hours:", student_df['smartphone_usage_hours'].describe())


# Data Preprocessing Phase

# 1. Dropping the Irrelevant Columns 

student_df.drop(['student_id'], axis=1, inplace=True)

# Handling Catorigal Variables 

categorical_cols = ['gender', 'mental_health_status', 'parent_education_level','internet_quality']

student_df = pd.get_dummies(student_df, columns=categorical_cols, drop_first=True)

# Features and Target

X = student_df.drop('final_exam_score', axis=1)

y = student_df['final_exam_score'] # Regression Problem


# Applying Decision Tree Regressor

# 1. Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Creating a Decision Tree Regressor

regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions on the test set

y_pred = regressor.predict(X_test)

# Evaluating the model

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R^2 Score: {r2:.4f}')


# Creating a Decision Tree Classifier


student_df['performance'] = pd.cut(student_df['final_exam_score'], bins=[0, 60, 75, 100], labels=['Poor', 'Average', 'Good'])

x_cls = student_df.drop(['final_exam_score', 'performance'], axis=1)
y_cls = student_df['performance']

X_train, X_test, y_train, y_test = train_test_split(x_cls, y_cls, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualizing the Decision Tree


# 1. Save the Regressor Tree
plt.figure(figsize=(20,10))
tree.plot_tree(regressor, filled=True, feature_names=X.columns, max_depth=3, fontsize=10)
plt.savefig('regressor_tree.png', dpi=300, bbox_inches='tight')
plt.close() # Closes the current figure so the next one starts fresh

# 2. Save the Classifier Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=x_cls.columns, class_names=list(clf.classes_), fontsize=10)
plt.savefig('classifier_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Using dtreeviz for a more advanced visualization (Regressor)


viz_model = dtreeviz.model(
    model=regressor,
    X_train=X,                          
    y_train=y,                      
    target_name='final_exam_score',
    feature_names=X.columns.tolist(),
)

viz = viz_model.view()
viz.save('dtree_regression_graphviz.svg')
print("Graphviz Visualization successfully saved as 'dtree_regression_graphviz.svg'")



# 3. Using dtreeviz for a more advanced visualization (classification)

viz_model = dtreeviz.model(
    model=clf,
    X_train=x_cls,                          
    y_train=y_cls,                      
    target_name='performance',
    feature_names=x_cls.columns.tolist(),
    class_names=list(clf.classes_),
)

viz = viz_model.view()
viz.save('dtree_classification_graphviz.svg')
print("Graphviz Visualization successfully saved as 'dtree_classification_graphviz.svg'")


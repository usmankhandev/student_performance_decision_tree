import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Setting the global font
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=== Student Performance - Decision Tree with Graphviz ===\n")

# Loading and Preprocessing the Dataset
student_df = pd.read_csv('student_digital_life.csv')
student_df.drop(['student_id'], axis=1, inplace=True)

categorical_cols = ['gender', 'mental_health_status', 'parent_education_level', 'internet_quality']
student_df = pd.get_dummies(student_df, columns=categorical_cols, drop_first=True)

# Regression Target
X = student_df.drop('final_exam_score', axis=1)
y = student_df['final_exam_score']

# Classification Target
student_df['performance'] = pd.cut(student_df['final_exam_score'], bins=[0, 60, 75, 100], labels=['Poor', 'Average', 'Good'])

X_cls = student_df.drop(['final_exam_score', 'performance'], axis=1)
y_cls = student_df['performance']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Implementing Models Both Classification and Regression
regressor = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42)
regressor.fit(X_train, y_train)

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
clf.fit(X_train_cls, y_train_cls)

# Evaluating both the models

y_pred_reg = regressor.predict(X_test)
print("\n=== REGRESSION RESULTS ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_reg)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_reg):.4f}")

y_pred_cls = clf.predict(X_test_cls)
print("\n=== CLASSIFICATION RESULTS ===")
print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print(classification_report(y_test_cls, y_pred_cls))

# Visualizing the Final Trees

# 1. Simple PNG Trees of both Regressor and CLF models.

plt.figure(figsize=(22, 12))
plot_tree(regressor, filled=True, feature_names=X.columns, max_depth=3, fontsize=9, rounded=True)
plt.title("Decision Tree Regressor (max_depth=3)")
plt.savefig('decision_tree_regressor.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(22, 12))
plot_tree(clf, filled=True, feature_names=X_cls.columns, 
          class_names=list(clf.classes_), max_depth=3, fontsize=9, rounded=True)
plt.title("Decision Tree Classifier")
plt.savefig('decision_tree_classifier.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Graphviz Export (SVG + DOT files)

# Regression Tree
dot_data_reg = export_graphviz(
    regressor, 
    out_file=None,
    feature_names=X.columns,
    filled=True, 
    rounded=True,
    special_characters=True,
    precision=2
)

with open("regressor_tree.dot", "w") as f:
    f.write(dot_data_reg)

# Classification Tree
dot_data_cls = export_graphviz(
    clf, 
    out_file=None,
    feature_names=X_cls.columns,
    class_names=list(clf.classes_),
    filled=True, 
    rounded=True,
    special_characters=True
)

with open("classifier_tree.dot", "w") as f:
    f.write(dot_data_cls)

print("\n✅ Graphviz files generated:")
print("   • regressor_tree.dot")
print("   • classifier_tree.dot")

# Try to convert DOT to SVG (if graphviz is installed)
try:
    import graphviz
    graph_reg = graphviz.Source(dot_data_reg)
    graph_reg.render("decision_tree_regressor", format='svg', cleanup=True)
    
    graph_cls = graphviz.Source(dot_data_cls)
    graph_cls.render("decision_tree_classifier", format='svg', cleanup=True)
    
    print("   • decision_tree_regressor.svg")
    print("   • decision_tree_classifier.svg")
    print("\n🎉 SVG files generated successfully!")
except Exception as e:
    print(f"\nNote: Could not auto-generate SVG. Install graphviz system package:")
    print("   sudo apt install graphviz    # (on Ubuntu/Linux)")
    print("   Then run: dot -Tsvg regressor_tree.dot -o regressor_tree.svg")

print("\nAll done! Check the folder for PNG, DOT, and SVG files.")
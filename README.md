# Data_Mining

## Dataset used: 
- [Banking Dataset](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)

## Libraries used:
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

## Algorithms used:
- KNN
- Naive Bayes
- Decision Tree
- Random Forest
- confusion matrix

## Functions used:
- `load_csv(file_path)`
- `determine_features_and_goal(df)`
- `split_data(features, goal, test_size=0.5, random_state=3)`
- `label_encode_categorical_features(df)` 
- `solve_missing(df)`
- `apply_knn_classifier(K,X_train, X_test, y_train)`
- `apply_naive_bayes_classifier(X_train, X_test, y_train)`
- `apply_decision_tree_classifier(X_train, X_test, y_train)`
- `apply_random_forest_classifier(X_train, X_test, y_train)`
- `calculate_performance(y_test, y_pred)`

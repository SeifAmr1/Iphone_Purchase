
iPhone Purchase Prediction - Phase 1

This project analyzes a consumer dataset to predict whether a user will purchase an iPhone based on demographic features using Machine Learning classifiers. The focus is on comparing the performance of Naive Bayes and K-Nearest Neighbors (KNN) classifiers with proper preprocessing, resampling, scaling, and hyperparameter tuning.

---

Project Workflow:

1. Data Preprocessing

  * Loaded `Iphone_purchase.csv` and dropped non-informative `User ID`.
  * Encoded the categorical `Gender` feature using `LabelEncoder`.

2. Class Balancing with SMOTE

  * Applied **SMOTE** to balance the training dataset due to class imbalance in the target (`Purchased`).

3. Feature Scaling

  * Used `StandardScaler` for Gaussian Naive Bayes (continuous features).
  * Used `MinMaxScaler` for KNN (to ensure all features are on the same scale).

---

Classifier 1: Hybrid Naive Bayes

  Separated features into:

  * Continuous: `Age`, `EstimatedSalary` → modeled using GaussianNB
  * Categorical: `Gender` → modeled using CategoricalNB
  
Performed GridSearchCV on:

   *`var_smoothing` for GaussianNB
   *`alpha` for CategoricalNB
   
 Final prediction is based on the product of probabilities from both models.
 Evaluation:

   * Accuracy, Confusion Matrix, and Classification Report printed.

---

Classifier 2: K-Nearest Neighbors (KNN)

 Normalized features using MinMaxScaler.
 Applied GridSearchCV on:

   * `n_neighbors` from 1 to 10
   * `weights`: `'uniform'` and `'distance'`
   * `metric`: `'euclidean'` and `'manhattan'`
   
 Selected best hyperparameters and tested final model on unseen test data.

---

Performance Comparison

 Compared KNN and Naive Bayes accuracies on the test set using a bar chart.

---

 Notes

* SMOTE improved the class balance and helped generalize better.
* Although manual KNN tuning gave slightly higher accuracy (\~95%), GridSearch was used for **scalability and generalization**.
* The combination of **hybrid Naive Bayes modeling** and **automated KNN tuning** offers insights into model suitability based on feature types and distribution.

---



iPhone Purchase Prediction - Phase 2

This phase extends the machine learning analysis by exploring advanced classifiers MLP (Neural Network), Support Vector Machine (SVM), and Decision Tree to predict whether a user will purchase an iPhone based on demographic data. The dataset is preprocessed and balanced, and extensive hyperparameter tuning is performed to optimize model performance.

---

Project Workflow:

1.Data Preprocessing

  * Loaded and cleaned the dataset (`Iphone_purchase.csv`), removing the `User ID` column.
  * Applied **OneHotEncoding** to the `Gender` feature (`Gender_Male`).
  * Split data into training and testing sets.
  * Balanced training data using **SMOTE** (Synthetic Minority Over-sampling Technique).

2. Feature Scaling

  * Used **MinMaxScaler** for `MLPClassifier` (neural networks are sensitive to scale).
  * Used **StandardScaler** for `SVM` (improves kernel convergence).
  * No scaling required for `Decision Tree` (tree-based models are not scale-sensitive).

---

Model 1: Multi-Layer Perceptron (MLP)

  * Tuned using `GridSearchCV` with 5-fold cross-validation on parameters:
  
    * `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`
  * Evaluated on both **normalized** and **standardized** data.
  * Output: best hyperparameters, accuracy, and predictions on the test set.

---

Model 2: Support Vector Machine (SVM)

  * Grid search for optimal:
  
    * `C` (penalty), `gamma` (kernel coefficient), `kernel` (`linear` or `rbf`)
  * Evaluated the effect of different `C` values using a plot.
  * Output: best parameters, test set predictions, and final accuracy.

---

Model 3: Decision Tree

  * Grid search tuning on:
  
    * `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
  * Also tested manually varying `max_depth` to observe accuracy impact.
  * No feature scaling required for decision trees.
  * Output: best parameters, predictions, and test accuracy.

---

Final Comparison

  * Bar chart comparing test accuracies of:
  
    * MLP
    * SVM
    * Decision Tree

---

Key Observations:

  * SMOTE ensured balanced class distribution for robust training.
  * MLP performance was sensitive to scaling and hyperparameters.
  * SVM required proper kernel and scaling choice for optimal results.
  * Decision Trees performed well without preprocessing but needed careful depth control to avoid overfitting.
  * GridSearchCV provided a systematic and scalable approach to finding optimal hyperparameters for all models.

---

 Requirements:

* `pandas`, `scikit-learn`, `imblearn`, `matplotlib`, `numpy`






extern crate rusticlearning;

use rusticlearning::models::{train_logistic_regression, train_knn, train_random_forest};
use linfa::prelude::*;
use linfa_datasets::iris;

#[test]
fn test_logistic_regression_training() {
    let (train, _) = iris().split_with_ratio(0.8);
    let model = train_logistic_regression(&train);
    assert!(model.is_ok(), "Failed to train Logistic Regression model");
}

#[test]
fn test_knn_training() {
    let (train, _) = iris().split_with_ratio(0.8);
    let model = train_knn(&train);
    assert!(model.is_ok(), "Failed to train KNN model");
}

#[test]
fn test_random_forest_training() {
    let (train, _) = iris().split_with_ratio(0.8);
    let model = train_random_forest(&train);
    assert!(model.is_ok(), "Failed to train Random Forest model");
}

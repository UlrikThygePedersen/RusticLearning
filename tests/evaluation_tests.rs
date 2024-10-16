extern crate rusticlearning;

use rusticlearning::models::train_logistic_regression;
use rusticlearning::evaluation::evaluate_model;
use linfa::prelude::*;
use linfa_datasets::iris;

#[test]
fn test_model_evaluation() {
    let (train, test) = iris().split_with_ratio(0.8);
    let model = train_logistic_regression(&train).unwrap();
    let result = evaluate_model(&model, &test);
    assert!(result.is_ok(), "Model evaluation failed");
}

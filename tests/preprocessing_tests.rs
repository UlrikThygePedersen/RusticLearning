extern crate rusticlearning;

use rusticlearning::preprocessing::preprocess_data;
use linfa::prelude::*;
use linfa_datasets::iris;

#[test]
fn test_data_preprocessing() {
    let (train, test) = iris().split_with_ratio(0.8);
    let result = preprocess_data(train, test);
    assert!(result.is_ok(), "Preprocessing failed");
    
    let (scaled_train, scaled_test) = result.unwrap();
    assert_eq!(scaled_train.records().shape()[1], 4, "Feature scaling went wrong");
    assert_eq!(scaled_test.records().shape()[1], 4, "Test data preprocessing failed");
}

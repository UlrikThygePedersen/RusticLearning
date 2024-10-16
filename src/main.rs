mod preprocessing;
mod models;
mod evaluation;
mod utils;

use models::{train_logistic_regression, train_knn, train_random_forest};
use preprocessing::preprocess_data;
use utils::load_data;
use evaluation::evaluate_models;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load and preprocess the data
    let (train, test) = load_data()?;
    let (train, test) = preprocess_data(train, test)?;

    // Train multiple models
    let logistic_model = train_logistic_regression(&train)?;
    let knn_model = train_knn(&train)?;
    let rf_model = train_random_forest(&train)?;

    // Evaluate and compare models
    evaluate_models(&logistic_model, &knn_model, &rf_model, &test)?;

    Ok(())
}

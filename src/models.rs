use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use linfa_knn::KNearestNeighbors;
use linfa_random_forest::RandomForest;
use std::error::Error;

pub fn train_logistic_regression(train: &Dataset<f64, usize>) -> Result<LogisticRegression<f64>, Box<dyn Error>> {
    // Train Logistic Regression model
    let model = LogisticRegression::default()
        .max_iterations(200)
        .fit(train)?;
    Ok(model)
}

pub fn train_knn(train: &Dataset<f64, usize>) -> Result<KNearestNeighbors<f64>, Box<dyn Error>> {
    // Train K-Nearest Neighbors (K=3)
    let model = KNearestNeighbors::new(3)
        .fit(train)?;
    Ok(model)
}

pub fn train_random_forest(train: &Dataset<f64, usize>) -> Result<RandomForest<f64>, Box<dyn Error>> {
    // Train Random Forest (10 trees)
    let model = RandomForest::params()
        .n_trees(10)
        .fit(train)?;
    Ok(model)
}

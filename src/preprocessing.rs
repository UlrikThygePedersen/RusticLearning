use linfa::prelude::*;
use linfa_preprocessing::scaling::StandardScaler;
use std::error::Error;

/// Function to preprocess training and test datasets
pub fn preprocess_data(train: Dataset<f64, usize>, test: Dataset<f64, usize>) -> Result<(Dataset<f64, usize>, Dataset<f64, usize>), Box<dyn Error>> {
    // Standardize the features (zero mean, unit variance)
    let scaler = StandardScaler::fit(&train)?;
    let train = scaler.transform(train);
    let test = scaler.transform(test);

    Ok((train, test))
}

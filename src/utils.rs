use linfa_datasets::iris;
use linfa::prelude::*;
use std::error::Error;

/// Load the Iris dataset and split it into training and test sets
pub fn load_data() -> Result<(Dataset<f64, usize>, Dataset<f64, usize>), Box<dyn Error>> {
    let dataset = iris();
    let (train, test) = dataset.split_with_ratio(0.8);
    Ok((train, test))
}

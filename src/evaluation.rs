use linfa::prelude::*;
use linfa::metrics::ConfusionMatrix;
use std::error::Error;

pub fn evaluate_models<L, K, R>(
    logistic_model: &L,
    knn_model: &K,
    rf_model: &R,
    test: &Dataset<f64, usize>
) -> Result<(), Box<dyn Error>>
where
    L: PredictRef<ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>, usize>,
    K: PredictRef<ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>, usize>,
    R: PredictRef<ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>, usize>,
{
    println!("Evaluating Logistic Regression:");
    evaluate_model(logistic_model, test)?;

    println!("\nEvaluating K-Nearest Neighbors:");
    evaluate_model(knn_model, test)?;

    println!("\nEvaluating Random Forest:");
    evaluate_model(rf_model, test)?;

    Ok(())
}

fn evaluate_model<M>(model: &M, test: &Dataset<f64, usize>) -> Result<(), Box<dyn Error>>
where
    M: PredictRef<ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>, usize>,
{
    // Predict on the test set
    let predictions = model.predict(test);

    // Compute confusion matrix
    let cm = predictions.confusion_matrix(test)?;
    let accuracy = cm.accuracy();
    let f1_score = cm.f1_score(Some(1));

    // Print performance metrics
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    println!("F1 Score: {:.2}%", f1_score.unwrap() * 100.0);

    Ok(())
}

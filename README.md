# RusticLearning <img src="mascot.svg" alt="Linfa" width="20"/>


<img src="assets/intersection.png" alt="Intersection Image"/>

This project is a comprehensive machine learning pipeline written in Rust. It demonstrates how to build a machine learning system with multiple models, data preprocessing, and evaluation capabilities, all orchestrated in a modular way. The project uses the `linfa` ecosystem to implement common machine learning algorithms such as **Logistic Regression**, **K-Nearest Neighbors**, and **Random Forest**.

## Features

- **Data Loading and Preprocessing**: Load the Iris dataset and preprocess it (scaling features).
- **Multiple Models**: Train and compare multiple machine learning models:
  - Logistic Regression
  - K-Nearest Neighbors
  - Random Forest
- **Evaluation and Metrics**: Evaluate the models on the test set using metrics such as:
  - Accuracy
  - F1 Score
  - Confusion Matrix
- **Orchestrator**: Orchestrates the entire pipeline from data loading, model training, evaluation, and comparison.

## Dependencies

The project uses the following Rust crates:
- [linfa](https://crates.io/crates/linfa): A Rust crate for machine learning.
- [linfa-logistic](https://crates.io/crates/linfa-logistic): Logistic Regression.
- [linfa-knn](https://crates.io/crates/linfa-knn): K-Nearest Neighbors.
- [linfa-random-forest](https://crates.io/crates/linfa-random-forest): Random Forest.
- [linfa-datasets](https://crates.io/crates/linfa-datasets): Common datasets such as Iris.
- [ndarray](https://crates.io/crates/ndarray): N-dimensional array library for Rust.

## Project Structure

```plaintext
.
├── src
│   ├── main.rs                // Main orchestrator logic
│   ├── models.rs              // Module for model implementations
│   ├── preprocessing.rs       // Data preprocessing logic
│   ├── evaluation.rs          // Evaluation metrics and analysis
│   └── utils.rs               // Helper utilities (data loading, etc.)
└── Cargo.toml                 // Dependencies and project metadata
```


## Usage

### Prerequisites

To run this project, you need to have Rust installed. If you don’t have Rust installed, you can install it from [here](https://www.rust-lang.org/tools/install).

### Running the Project

1. **Clone the Repository**:

```bash
git clone https://github.com/your-username/rust-ml-orchestrator.git
cd rust-ml-orchestrator
```

2. **Run the Project**:

```bash
cargo run
```

The project will load the Iris dataset, preprocess it, train three different models, and evaluate their performance.

### Example Output

```plaintext
Evaluating Logistic Regression:
Accuracy: 96.67%
F1 Score: 96.55%

Evaluating K-Nearest Neighbors:
Accuracy: 96.67%
F1 Score: 96.55%

Evaluating Random Forest:
Accuracy: 100.00%
F1 Score: 100.00%
```

### Code Example

If you'd like to add more models or modify the evaluation process, here's an example of how to add a new model and use it in the orchestrator.

#### Add a New Model to `models.rs`

```rust
pub fn train_svm(train: &Dataset<f64, usize>) -> Result<YourSVMModel, Box<dyn Error>> {
    // Implement your SVM model training here
}
```

#### Use the New Model in `main.rs`

```rust
let svm_model = train_svm(&train)?;
evaluate_model(&svm_model, &test)?;
```

### Extending the Project

This project can be extended by:
- **Adding more machine learning models** from the `linfa` ecosystem.
- **Implementing more complex preprocessing** steps, such as feature selection, normalization, or handling missing data.
- **Customizing evaluation metrics** to include precision, recall, or ROC-AUC for binary classifiers.

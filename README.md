# MLproject

## Requirements
- **Python 3.8+**
- **Libraries**:
  - `pandas`
  - `numpy`
  - `torch`
  - `sklearn`
  - `matplotlib`
  - `optuna`
- **Hardware**: GPU support (optional, for faster training with CUDA).


### Installation
1. Clone or download the project to your local machine:


3. Ensure the data files (`margednew.csv`, `input_test.csv`, `output_test.csv`) and the saved model (`soc_lstm_model_new.pth`) are in the project directory.

## Usage

### 1. Preprocess Data
The dataset is preprocessed to normalize features and create sequences for LSTM. Run the initial data preparation (if not already done):
- Ensure `margednew.csv` is present and contains the required columns (`Voltage [V]`, `Current [A]`, `Temperature [degC]`, `Capacity [Ah]`, `Cumulative_Capacity_Ah`, `SOC [-]`).
- The script automatically generates `normalized_margednew.csv`.

### 2. Train the Model
The LSTM model is trained with hyperparameter optimization using Optuna. To retrain (optional):
- Run the training section of the notebook or modify `plot_mae_lstm.py` to include training logic.
- The trained model is saved as `soc_lstm_model_new.pth`.

### 3. Plot Training MAE
To visualize the training and validation Mean Absolute Error (MAE):
- Run `plot_mae_lstm.py`:
- Output: `training_validation_mae_lstm.png` showing MAE trends over 20 epochs.

### 4. Test the Model
To evaluate the model on new data and compare with actual values:
- Run `test_input_output_scatter.py`:


- Inputs: `input_test.csv` (features) and `output_test.csv` (actual SOC).
- Outputs:
- `actual_vs_predicted_scatter_input_test.png`: Scatter plot of Predicted vs. Actual SOC.
- Console output: MAE, MSE, and sample predictions/actual values.

### 5. Analyze Results
- Check the generated plots for model performance.
- Use MAE and MSE to assess prediction accuracy.
- Adjust hyperparameters or data if needed based on results.

## Data Description
- **margednew.csv**: Raw dataset with battery features and SOC labels.
- **input_test.csv**: Test dataset with features for prediction.
- **output_test.csv**: Test dataset with actual SOC values for comparison.

## Results
- **Training**: The model converges after 20 epochs with a test MAE of approximately 0.0129 and MSE of 0.0006958.
- **Testing**: On `input_test.csv`, the model achieves an MAE of ~0.0169 and MSE of ~0.0009107, with a scatter plot showing reasonable alignment with actual values.

## Future Work
- Extend the model to predict other battery parameters (e.g., State of Health).
- Incorporate real-time data streaming for live predictions.
- Optimize for deployment on edge devices.

## Contributing
Feel free to fork this repository, submit issues, or pull requests for improvements. Contact the author at [your-email] for collaboration.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if applicable, create a `LICENSE` file).

## Acknowledgments
- Thanks to xAI for providing tools and inspiration.
- Credit to the open-source community for libraries like PyTorch and Optuna.

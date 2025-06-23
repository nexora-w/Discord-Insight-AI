# Discord Recruitment Sentence Classifier

This project is a machine learning pipeline for classifying recruitment-related sentences, with a Flask API for inference.

## Project Structure

```
.
├── app/                # Flask API for model inference
├── data/               # Raw and processed data
├── model/              # Saved model and vectorizer
├── test/               # Test scripts and test data
├── requirements.txt    # Python dependencies
├── train_model.py      # Script to train the model
```

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Train the model:
   ```bash
   python train_model.py
   ```
4. Run the API:
   ```bash
   python app/main.py
   ```

## Usage

- **Training:**
  - Edit `train_model.py` or replace `data/recruitment_sentences.csv` with your own data.
  - Run the script to train and save the model and vectorizer.
- **API:**
  - Start the Flask app and send a POST request to `/predict` with a JSON body:
    ```json
    { "sentence": "Your sentence here" }
    ```
- **Testing:**
  - Use `test/index.py` to evaluate the model on test data in `test/test.csv`.

## Requirements
- Python 3.7+
- Flask
- pandas
- scikit-learn
- joblib

## Reports, plots, metrics

After training the model with `train_model.py`, the following evaluation outputs are generated:

- **Classification Report:** Printed in the console after training, showing precision, recall, f1-score, and support for each class.
- **Accuracy on Test Set:** Run `python test/test_model.py` to print the accuracy on your test data.
- **Saved Reports and Plots:**
  - A text file with the classification report and a PNG image of the confusion matrix will be saved in the `reports/` directory after training (see code in `train_model.py`).

### Example commands

```bash
python train_model.py
python test/test_model.py
```

Check the `reports/` directory for saved evaluation outputs.

## License
MIT License (add LICENSE file if needed) 
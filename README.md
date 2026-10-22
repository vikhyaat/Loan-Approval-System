# Loan Approval Prediction — Streamlit App

A runnable Streamlit web app based on the columns and labels in the supplied `loan_approval_data.csv` dataset. It uses a Gaussian Naive Bayes classifier and provides a prediction form, dataset summary, and evaluation metrics.

## Run on Windows (VS Code)

1. Extract the ZIP file to a folder.
2. Open that folder in VS Code.
3. Open **Terminal → New Terminal**.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

7. Open the local address Streamlit prints (normally http://localhost:8501).

## Included files

- `app.py` — Streamlit app; trains the model from the CSV on first launch.
- `loan_approval_data.csv` — dataset supplied with your notebook.
- `requirements.txt` — Python dependencies.

## Notes

- Keep `app.py` and `loan_approval_data.csv` in the same folder.
- The app excludes rows whose `Loan_Approved` label is missing from training.
- The app is an educational demo, not a real bank decision tool.

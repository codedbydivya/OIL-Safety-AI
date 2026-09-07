OIL Safety AI MVP — SIH 2026 PS 26165

1. Install Python 3.11 or newer.
2. Open this folder in VS Code.
3. Open the VS Code terminal.
4. Run:
   python -m venv venv
5. Activate:
   Windows: venv\Scripts\activate
   macOS/Linux: source venv/bin/activate
6. Install:
   pip install -r requirements.txt
7. Train:
   python train_model.py
8. Run the dashboard:
   streamlit run app.py

Important:
- data/reports.csv is DEMO/SYNTHETIC data only.
- Replace it with authorized OIL data when available.
- This MVP uses TF-IDF + Logistic Regression for the first working model.
- The rule tagging is a transparent keyword-based prototype.
- For the SIH final version, improve the dataset, labels, validation, explainability and domain-specific NLP.

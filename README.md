# Sports Betting Predictor (Football/Soccer)

## [👉 LIVE PREDICTION SITE 👈](https://aisportbetter.com)

---

## Description
This project is designed to help you spot value bets—those with a positive expected return. The idea is: by looking at how betting odds change over time, the system learns to predict the true chances of game outcomes. When the odds on offer deviate from these predictions, it might signal a smart betting opportunity.


The site displays current profitable betting opportunities identified by our model based on real-time data analysis.

## Machine Learning Model
Our predictive models have been rigorously developed and refined over time:
- Trained on multiple years of historical betting data
- Extensive architecture testing (LSTM, Transformers, and traditional ML models)
- Hyperparameter optimization through cross-validation
- Continuous retraining with new data to adapt to market changes and newer data

The final deployed model outperformed all benchmarks in backtesting and continues to show strong performance in live prediction scenarios.

## How It Works
1.  **Data Scraping:** Sports betting data, including odds, is scraped automatically every 15 minutes.
2.  **Model Training:** A machine learning model (LSTM) has been trained on years of historical scraped data to learn prediction patterns.
3.  **Live Predictions:** The trained model runs continuously in production, analyzing the latest data to identify potentially profitable betting opportunities, which are then displayed on the website ([aisportbetter.com](https://aisportbetter.com)).

All backend processes (scraping, ETL, inference, database) run continuously on AWS to ensure you always have access to the most current betting opportunities displayed on the GCP-hosted site.

---

# How to Run the Scraper Locally

Follow these steps to get the scraper up and running on your local machine.

1. Clone the repository by opening a terminal and running:
```bash
git clone https://github.com/BoazBD/Winner-Predictor.git
cd Winner-Predictor
```
2. Create a virtual environment and installing requirements (for macOS/Linux):
```bash
python -m venv venv
source venv/bin/activate
```
3. Install the requirements:
```bash
pip install -r requirements.txt
```
4. Run the scraper using:
```bash
python scraper/main.py
```
You should now see the scraper running in your terminal, and the scraped data should be saved in a file called bets.csv

*Important Note:* 

This scraper must be run with an Israeli IP address to access the target sports betting website.

## Examples of the data visualized
  
<img width="1440" alt="Screenshot 2025-03-14 at 20 40 32" src="https://github.com/user-attachments/assets/066b29dd-a893-416e-9e56-4632bbf7919e" />
<img width="1443" alt="Screenshot 2025-03-14 at 20 41 08" src="https://github.com/user-attachments/assets/d07c6c83-d6f9-4a34-ae23-cd4c07ff6296" />
<img width="1443" alt="Screenshot 2025-03-14 at 20 41 39" src="https://github.com/user-attachments/assets/f7a07fd7-9178-4e94-9299-8484d18feb2f" />

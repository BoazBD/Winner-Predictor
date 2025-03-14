# Sports Betting Predictor

---

## Description
This project is designed to help you spot value bets—those with a positive expected return. The idea is: by looking at how betting odds change over time, the system learns to predict the true chances of game outcomes. When the odds on offer deviate from these predictions, it might signal a smart betting opportunity.

## How It Works
1. **Automated Scraping:** Sports data, including the continuously updated odds, is scraped every 15 minutes from a betting website.
2. **ETL Processing:** The scraped data is then processed, where it undergoes cleaning, transformation, and loading into a data warehouse.
3. **Model Training:** An optimized LSTM model is trained on historical data. The model learns to predict outcome probabilities based on past trends in the changing odds.
4. **Prediction Generation:** The model generates predictions for upcoming games. When the site's odds differ significantly from the model's predictions, the system flags these as potential value bets.
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

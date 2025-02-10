# E-Commerce Competitor Strategy Dashboard

This project provides a real-time dashboard that helps e-commerce businesses analyze competitor data, customer sentiment, and forecast pricing strategies. By using various machine learning models and APIs, the dashboard generates actionable strategic recommendations to optimize pricing, promotions, and customer satisfaction.

## Features

- Load and analyze competitor data (prices, discounts, etc.).
- Perform sentiment analysis on product reviews.
- Forecast future discounts using ARIMA.
- Generate strategic recommendations using a large language model (LLM).
- Display competitor data and sentiment analysis results on an interactive dashboard.

## Prerequisites

Before running the project, make sure you have the following installed:

- Python 3.x
- pip (Python package installer)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sahilmate/e-commerce-competitor-analysis.git
cd e-commerce-competitor-analysis
```

### 2. Install dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

Create a `.env` file in the root directory of the project and add your sensitive API keys and webhook URL.

```env
GROQ_API_KEY="your_groq_api_key"
SLACK_WEBHOOK_URL="your_slack_webhook_url"
```

### 4. Running the project

After setting up the environment, you can run the project by using the following command:

```bash
streamlit run competitor_strategy_dashboard.py
```

This will launch the Streamlit app, and you can open the dashboard in your browser.

## Project Structure

```
e-commerce-competitor-strategy-dashboard/
├── competitor_strategy_dashboard.py # Main Streamlit application file
├── .env                             # File to store your API keys and webhook URLs
├── requirements.txt                 # Python dependencies
├── price_data.csv                   # CSV file with competitor price data
├── review_data.csv                  # CSV file with customer review data
├── webscraping.ipynb                # Jupyter notebook file for web scraping 
```

## Dependencies

Here are the Python libraries required for the project:

- `json`
- `pandas`
- `requests`
- `numpy`
- `plotly`
- `streamlit`
- `openai`
- `sklearn`
- `statsmodels`
- `transformers`
- `dotenv`
- `selenium`
- `webdriver_manager`

You can install all the dependencies using:

```bash
pip install -r requirements.txt
```

## Requirements.txt

```txt
pandas==1.5.3
numpy==1.23.4
requests==2.28.1
plotly==5.10.0
streamlit==1.15.2
openai==0.27.0
sklearn==1.1.2
statsmodels==0.13.5
transformers==4.28.0
python-dotenv==0.21.1
selenium==4.8.1
webdriver_manager==3.8.5
```

## API Keys and Webhooks

The project requires two key components to work:

1. **GROQ API Key**: Used to generate strategic recommendations based on competitor data, sentiment, and forecasting. You can obtain your API key from [Groq](https://groq.com/).

2. **Slack Webhook URL**: Used to send the strategic recommendations directly to a Slack channel. You can create a Slack Webhook URL [here](https://api.slack.com/messaging/webhooks).

Make sure to store these values securely in the `.env` file, as shown below:

```env
GROQ_API_KEY=your_groq_api_key
SLACK_WEBHOOK_URL=your_slack_webhook_url
```
## Slack Webhook Image:
![Slack Webhook Image](https://github.com/sahilmate/e-commerce-competitor-analysis/blob/main/Slack%20Webhook.jpeg)

## Streamlit Dashboard Images: 
![Streamlit Dashboard](https://github.com/sahilmate/e-commerce-competitor-analysis/blob/main/Streamlit%20Dashboard-1.jpeg)
![Streamlit Dashboard](https://github.com/sahilmate/e-commerce-competitor-analysis/blob/main/Streamlit%20Dashboard-2.jpeg)


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







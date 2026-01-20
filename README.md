# Dendritic LSTM Stock Prediction Model Attempt

## Intro

Description:

This project attempts to create a Dendritic LSTM model for stock prediction. The `download.py` script dynamically creates the dataset used for this project. The download script attempts to get as much data as possible from companies traded on the NASDAQ (The script consistently gets ~3k companies in around ~30s). Unfortunately, due to starting this project on Saturday (1/17/26), no hyperparameter sweeps were performed.

Team:

- Vishwesswaran Gopal - First Year Engineering Student @ Purdue - [Linkedin](https://www.linkedin.com/in/vishwesswaran-gopal-76403826a/) - gopal21@purdue.edu

## Project Impact

LSTM models are commonly used for natural language tasks, such as machine translation and sentiment analysis; speech and audio processing, like speech and command recognition; time series forecasting, such as financial and weather forecasting; anomaly detection, i.e., fraud detection, and many more use cases not listed here.

Due to the importance and abundance of LSTM models, assessing whether LSTM models perform better under dendritic optimization remains crucial as dendritic optimization may allow these models to become cheaper and more efficient, thereby saving money, resources, and time.

Thus, this projects attempts to apply dendritic optimization to to a common use case of LSTM models, stock prediction, to determine the viability of dendritic LSTM models.

## Usage Instructions

Installation:

    git clone https://github.com/VG-Fish/Dendritic-Stock-Prediction-Model.git
    cd "Dendritic-Stock-Prediction-Model"
    pip install uv
    uv sync --locked

Run:

```python
# For dendritic LSTM model
uv run dendritic_main.py

# For normal LSTM model
uv run main.py

# To get more info about the parameters you can pass into the main scripts, run:
uv run <CHOOSE_MAIN>.py --help
```

## Results

Both models were trained on all the CSVs present in the `/stocks` folder of the generated dataset. In total, there were 2114 CSV files. The generated dataset also contains a `/etfs`, but no ETF data was used. Before the models are trained, all the CSVs are parsed and sequences of `SEQUENCE_LENGTH` (default is 30 days, can be changed by passing in `--sequence_length <YOUR_NUM_HERE>`) are generated from only the `Close` column, which represents a stock's closing price for a particular day. The models are fed sequences and aim to predict the price for the next day.

Comparing the traditional model to the dendritic model below:

| Model       | Final Validation Score (MSE) |
| ----------- | ---------------------------- |
| Traditional | 0.8203176259994507           |
| Dendritic   | 0.8195596933364868           |

This provides a Remaining Error Reduction of **0.092%**.

## Raw Results Graph

![Example Perforated AI output graph.](model_info/model_info.png)

## Resources Used

- [Stock Price Prediction in Python with PyTorch - Full Tutorial ](https://www.youtube.com/watch?v=IJ50ew8wi-0)
- [Download NASDAQ Historical Data Reference Script](https://www.kaggle.com/code/jacksoncrow/download-nasdaq-historical-data)
- [Perplexity](https://www.perplexity.ai/)

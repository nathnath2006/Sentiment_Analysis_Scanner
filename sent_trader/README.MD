The purpose of this readme is to explain the purpose of this project and to explain how to use it.

                                                     --- SUMMARY ---
This project implements an end-to-end data pipeline for collecting stock market data and associated news articles, performing NLP sentiment analysis, and storing everything in a clean Postgres database for further analysis. The system supports correlation analysis, visualization, and can serve as the backend for machine learning experiments or dashboards.
                                                    ------------------


The objective of this code was to be able for me to gather data about stock price and news in order to :
- Build a cohesive and clean database for further trainings of AI models
- Be able to explore correlation between price and news, using simple sentiment analysis

Scrape → Clean → Preprocess → Sentiment → Store (1) → Query → Visualize

(1) You have the option here to save as .csv for exports

The sentiment analysis here is handled by the lightweight VADER. Why not Finbert ? because VADER is fast and efficient and as this IS NOT made for trading purposes. For top results and to backtest or build strategies, Finbert would be the go-to.

Before launching the code :
- Set up your own database using any GUI you want, I used PgAdmin4. You'll find a empty, fresh, backup with the project just import it and run as a query. You are done !
![Database Diagram](images/db_diagram.png)
- Set up your .env -> Make sure that your own credential are set for the server.
- Look the requirment file for any missing modules and take a read at the notebook first before launching. (use pip install -r requirments.txt)

Utils.py contains all the helper functions, you don't need to change anything there but feel free to look and modify (if you know what you are doing).

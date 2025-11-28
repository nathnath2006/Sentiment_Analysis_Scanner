import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

#get connection through environment variables
#make sure to update the .env with your own credentials before running anything
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


#Utils for scraping and preprocessing news data
#Scrape_news builds a dictionnary of the news title concerning an inputed ticker
#Scrapped news spans over a dynamic range for now, based now news abundance (automatically calculated by Google)
def scrape_news(ticker):
    articles = []
    feed = feedparser.parse(f'https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en')

    for entry in feed.entries:
         articles.append({
            'title': entry.title,
            'ticker': ticker,
            'publish_date': entry.get('published', None),
            'link': entry.link
        })
    return articles


#Clean_html cleans any html anchors from the titles, just in case
#It takes into input a dataframe and returns a dataframe
def clean_html(news_data):
    cleaned_news = []
    for news in news_data['title']:
        soup = BeautifulSoup(news,'html.parser')
        cleaned_news.append(soup.get_text(news))
    news_data['title'] = cleaned_news
    return news_data

#preprocess_text has two functions :
# - it stems and tokenize the title for efficient sentiment analysis
# - it convert dates into a standardized python format
#It takes into input a dataframe and returns a dataframe
def preprocess_text(news_data):
    stemmer = PorterStemmer()
    preprocessed_title = []
    preprocessed_date =[]
    for news in news_data['title']:
        tokens = word_tokenize(news)
        stemmed_tokens = [stemmer.stem(w) for w in tokens]
        processed_text = ' '.join(stemmed_tokens)
        preprocessed_title.append(processed_text)
    news_data['title'] = preprocessed_title
    for date in news_data['publish_date']:
        py_date = datetime.strptime(date,  '%a, %d %b %Y %H:%M:%S %Z')
        preprocessed_date.append(py_date)
    news_data['publish_date'] = preprocessed_date
    return news_data

#calculate sentiment gives back the sentiment score of each title, the compound score:
# - => 0.05 means positive
# - =< 0.05 means negative
# - inbetween means neutral
#It takes into input a dataframe and returns a dataframe
def calculate_sentiment(news_data):
    sia = SentimentIntensityAnalyzer()
    sent_score = []
    for news in news_data['title']:
        score = sia.polarity_scores(news)
        sent_score.append(score['compound'])
    news_data['compound_sentiment_score'] = sent_score
    return news_data

#feed_databes stores any newly fetched data into the Postgres database
#It takes as input the news dataframe, the stock dataframe and the ticker symbol
#The specificity here it that it avoids duplicates by using the ON CONFLICT DO NOTHING clause, so no need worry about duplicates
def feed_database(news_data, stock_data, ticker):
    conn = get_connection()

    cur = conn.cursor()
    
    cur.execute("""
                INSERT INTO stock_list (symbol)
                VALUES (%s)
                ON CONFLICT (symbol) DO NOTHING;
            """, (ticker,))

    cur.execute("SELECT stock_id FROM stock_list WHERE symbol = %s;", (ticker,))
    stock_id = cur.fetchone()[0]

    for _, news_row in news_data.iterrows():
        cur.execute("""
            INSERT INTO articles (link, title, publish_date, stock_id, compound_sentiment_score)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (link) DO NOTHING;
        """, (news_row['link'], news_row['title'], news_row['publish_date'], stock_id, news_row['compound_sentiment_score']))
    
    for _, price_row in stock_data.iterrows():
        cur.execute("""
            INSERT INTO daily_stock_price(stock_ID, open, high, low, close, volume, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, stock_ID) DO NOTHING;
        """, (stock_id, price_row['Open'], price_row['High'], price_row['Low'], price_row['Close'], price_row['Volume'], price_row['Datetime']))

    conn.commit()
    cur.close()
    conn.close()
    return True


def save_database_to_csv():
    conn = get_connection()

    stock_df = pd.read_sql("""
        SELECT 
            s.symbol,
            p.date,
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume
        FROM daily_stock_price p
        JOIN stock_list s
            ON s.stock_id = p.stock_id;
    """, conn)

    articles_df = pd.read_sql("""
        SELECT
            s.symbol,
            a.link,
            a.title,
            a.publish_date,
            a.compound_sentiment_score
        FROM articles a
        JOIN stock_list s
            ON s.stock_id = a.stock_id;
    """, conn)

    conn.close()
    return stock_df, articles_df

# Get all tickers stored in the database
def get_stock_list():
    conn = get_connection()
    df = pd.read_sql("SELECT symbol FROM stock_list ORDER BY symbol ASC;", conn)
    conn.close()
    return df['symbol'].tolist()

# Get daily prices for a given ticker
def get_stock_prices(ticker):
    conn = get_connection()
    query = """
        SELECT p.date, p.open, p.high, p.low, p.close, p.volume
        FROM daily_stock_price p
        JOIN stock_list s ON p.stock_id = s.stock_id
        WHERE s.symbol = %s
        ORDER BY p.date ASC;
    """
    df = pd.read_sql(query, conn, params=(ticker,))
    conn.close()
    return df

# Get article sentiment for a given ticker
def get_article_sentiment(ticker):
    conn = get_connection()
    query = """
        SELECT a.publish_date, a.title, a.compound_sentiment_score
        FROM articles a
        JOIN stock_list s ON s.stock_id = a.stock_id
        WHERE s.symbol = %s
        ORDER BY a.publish_date ASC;
    """
    df = pd.read_sql(query, conn, params=(ticker,))
    conn.close()
    return df


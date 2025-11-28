import sqlite3
import pandas as pd

# we initialize the SQLite database

def init_db():
    conn = sqlite3.connect("sentiment.db")
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS stock_list (
            stock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS articles (
            article_id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            publish_date TEXT NOT NULL,
            stock_id INTEGER NOT NULL,
            compound_sentiment_score REAL,
            FOREIGN KEY (stock_id) REFERENCES stock_list(stock_id)
        );

        CREATE TABLE IF NOT EXISTS daily_stock_price (
            dsp_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            open REAL NOT NULL,
            close REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            volume INTEGER NOT NULL,
            date TEXT NOT NULL,
            UNIQUE(date, stock_id),
            FOREIGN KEY (stock_id) REFERENCES stock_list(stock_id)
        );
    """)

    conn.commit()
    conn.close()
    return True

# -------------------------------
# ALL THE QUERIES BELOW THIS LINE
# -------------------------------

# All the adding queries 
#query to add a stock symbol to the stock_list table
def add_stock(symbol):
    conn = sqlite3.connect("sentiment.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO stock_list (symbol)
        VALUES (?)
    """, (symbol,))
    conn.commit()
    conn.close()

#query to add daily prices for a stock to the daily_stock_price table
def add_daily_price(ticker, stock_data):
    conn = sqlite3.connect("sentiment.db")
    cur = conn.cursor()

    cur.execute("SELECT stock_id FROM stock_list WHERE symbol = ?;", (ticker,))
    stock_id = cur.fetchone()[0]

    for _, price_row in stock_data.iterrows():
        cur.execute("""
            INSERT INTO daily_stock_price(stock_ID, open, high, low, close, volume, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (date, stock_ID) DO NOTHING;
        """, (stock_id, price_row['Open'], price_row['High'], price_row['Low'], price_row['Close'], price_row['Volume'], price_row['Date'].isoformat()
))

    conn.commit()
    conn.close()

#query to add news articles for a stock to the articles table
def add_news_data(ticker, news_data):
    conn = sqlite3.connect("sentiment.db")
    cur = conn.cursor()

    cur.execute("SELECT stock_id FROM stock_list WHERE symbol = ?;", (ticker,))
    stock_id = cur.fetchone()[0]

    for _, news_row in news_data.iterrows():
        cur.execute("""
            INSERT INTO articles (link, title, publish_date, stock_id, compound_sentiment_score)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (link) DO NOTHING;
        """, (news_row['link'], news_row['title'], news_row['publish_date'].isoformat(), stock_id, news_row['compound_sentiment_score']))

    conn.commit()
    conn.close()


#All the fetching queries
#Save the entire database into two dataframes and return them for a csv export
def save_database_to_csv():
    conn = sqlite3.connect("sentiment.db")

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

#returns a list of all stock symbols in the stock_list table
def get_stock_list():
    conn = sqlite3.connect("sentiment.db")
    df = pd.read_sql("SELECT symbol FROM stock_list ORDER BY symbol ASC;", conn)
    conn.close()
    return df['symbol'].tolist()

# Get daily prices for a given ticker
def get_stock_prices(ticker):
    conn = sqlite3.connect("sentiment.db")
    query = """
        SELECT p.date, p.open, p.high, p.low, p.close, p.volume
        FROM daily_stock_price p
        JOIN stock_list s ON p.stock_id = s.stock_id
        WHERE s.symbol = ?
        ORDER BY p.date ASC;
    """
    df = pd.read_sql(query, conn, params=(ticker,))
    conn.close()
    return df

# Get article sentiment for a given ticker
def get_article_sentiment(ticker):
    conn = sqlite3.connect("sentiment.db")
    query = """
        SELECT a.publish_date, a.title, a.compound_sentiment_score
        FROM articles a
        JOIN stock_list s ON s.stock_id = a.stock_id
        WHERE s.symbol = ?
        ORDER BY a.publish_date ASC;
    """
    df = pd.read_sql(query, conn, params=(ticker,))
    conn.close()
    return df
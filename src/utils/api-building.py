#import needed libraries 
from fastapi import FastAPI, Query, Path,HTTPException
from typing import Optional
from datetime import date
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

#load environment variables
load_dotenv()

PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

# Create SQLAlchemy engine 
connection_string = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
print("Connection psql string:", connection_string)

engine = create_engine(connection_string, pool_pre_ping=True)

app= FastAPI(
    title="Malawi Stock Exchange (MSE) Daily Prices API",
    description="An API to access daily stock prices from the Malawi Stock Exchange (MSE).",
    version="1.0.0",
)

company_sector_mapping = {
    'AIRTEL': 'Telecommunication',
    'BHL': 'Hospitality',
    'FDHB': 'Finance',
    'FMBCH': 'Finance',
    'ICON': 'Construction',
    'ILLOVO': 'Agriculture',
    'MPICO': 'Construction',
    'NBM': 'Finance',
    'NBS': 'Finance',
    'NICO': 'Finance',
    'NITL': 'Finance',
    'OMU': 'Finance',
    'PCL': 'Investments',
    'STANDARD': 'Finance',
    'SUNBIRD': 'Hospitality',
    'TNM': 'Telecommunication'
}

# Home end-point
@app.get("/")
def home():
    return {"message": "Welcome to the MSE API!"}

#First end-point: Call with Query parameters
@app.get("/companies")
def get_companies(sector: Optional[str] = Query(default=None, description="Filter companies by sector")):
    """Get list of companies with optional sector filtering."""
    query = text("""
    SELECT *
    FROM tickers t
    JOIN daily_prices p ON t.counter_id = p.counter_id
    """)
    df = pd.read_sql(query, con=engine)
    df['Sector'] = df['ticker'].map(company_sector_mapping)
    df=df[['ticker','name','Sector','date_listed']]
    # Filter by sector if provided
    if sector:
        df = df[df['Sector'] == sector]
    data = df.to_dict(orient='records')
    return {'count': len(df), 'data': data}

#Second end-point: Call with Path parameters
@app.get("/companies/{ticker}")
def get_company_details(ticker:str):
    """Get company details and total records for a given ticker symbol."""
    #ticker details
    query = text("""
    SELECT t.ticker, t.name, t.date_listed, p.counter_id
    FROM tickers t
    JOIN daily_prices p ON t.counter_id = p.counter_id
    WHERE t.ticker = :ticker
    """)
    df = pd.read_sql(query, con=engine, params={"ticker": ticker})
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No company found for ticker '{ticker}'.")
    df['Sector'] = df['ticker'].map(company_sector_mapping)
    df = df[df['ticker'] == ticker]
    df=df[['ticker','name','Sector','date_listed']]
    company_details = df[['ticker', 'name', 'Sector', 'date_listed']].drop_duplicates().to_dict(orient='records')
    total_records = len(df)
    return {'Company details':company_details,'Total records':total_records} 

#Third end-point
@app.get("/prices/daily/range")
def get_daily_prices_by_date_range(
    ticker: str = Query(..., description="Stock ticker symbol"),
    start_date: Optional[date] =  Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(default=None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(default=100, description="Maximum records to return")):
    """ Get daily price data for a given ticker symbol within an optional date range and limit."""
    #fetch counter name from counter table
    query = text("""
    SELECT  p.trade_date, p.open_mwk, p.high_mwk, p.low_mwk, p.close_mwk, p.volume
    FROM tickers t
    JOIN daily_prices p ON t.counter_id = p.counter_id
    WHERE t.ticker = :ticker
    """)
    df = pd.read_sql(query, con=engine, params={'ticker': ticker})
    df.rename(columns={
    'open_mwk': 'open',
    'high_mwk': 'high',
    'low_mwk': 'low',
    'close_mwk': 'close',
    'volume': 'volume',
    'trade_date': 'trade_date'
    }, inplace=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # Filter by date range
    
    if start_date:
        df = df[df['trade_date'] >=start_date]
    if end_date:
        df = df[df['trade_date'] <=end_date]

    # Apply limit (max 1000)
    limit = min(limit or 100, 1000)
    df = df.head(limit)
    df = df.fillna('') 
    return {"Company": ticker, "data":df.to_dict(orient='records')}

#Fourth end-point
@app.get("/prices/daily/period")
def get_daily_prices_by_period(
    ticker: str = Query(..., description="Stock ticker symbol"),
    year: int = Query(..., description="Year"),
    month: Optional[int] = Query(default=None, description="Month of the year"),
):
    """Get daily price data for a given ticker symbol within a specified year and optional month."""
    
    query = text("""
        SELECT p.trade_date, p.open_mwk, p.high_mwk, p.low_mwk, p.close_mwk, p.volume
        FROM tickers t
        JOIN daily_prices p ON t.counter_id = p.counter_id
        WHERE t.ticker = :ticker
    """)
    
    df = pd.read_sql(query, con=engine, params={"ticker": ticker})
    df.columns = ['Period', 'Open', 'High', 'Low', 'Close', 'TotalVolume']
    df['Period'] = pd.to_datetime(df['Period'])
    
    df = df[df['Period'].dt.year == year]
    if month:
        df = df[df['Period'].dt.month == month]
    
    df['Period'] = df['Period'].dt.strftime('%Y-%m-%d')
    df = df.fillna('')
    
    return {"Company": ticker, "data": df.to_dict(orient='records')}

#Fith end-point
@app.get("/prices/latest")
def get_recent_prices(
    ticker: Optional[str] = Query(default=None, description="Stock ticker symbol"),
    ):
    """ Get the most recent price data for a given ticker symbol, including latest price,
    previous price, and percentage change."""
     #fetch counter name from counter table
    query = """
    SELECT t.ticker,p.trade_date, p.open_mwk, p.high_mwk, p.low_mwk, p.close_mwk, p.volume
    FROM tickers t
    JOIN daily_prices p ON t.counter_id = p.counter_id
    """
    df = pd.read_sql(query, con=engine)
    if ticker:
        df=df[df['ticker']==ticker]
   
    df.columns=[ticker,'trade_date','open',' high','low','close','Total Volume']
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df_sorted = df.sort_values(by='trade_date', ascending=False).reset_index(drop=True)

    # Latest price
    latest = df_sorted.iloc[0]
    latest_date = latest['trade_date']
    latest_price = latest['close']
    
    # Previous price
    if len(df_sorted) > 1:
        prev_price = df_sorted.iloc[1]['close']
        change = latest_price - prev_price
        change_percentage = (change / prev_price) * 100 if prev_price != 0 else 0
    else:
        prev_price = None
        change = None
        change_percentage = None
    if ticker:
        
        return {
            "ticker": ticker,
            "latest_date": latest_date,
            "latest_price": latest_price,
            "previous_price": prev_price,
            "change": change,
            "change_percentage": str(round(change_percentage,3))+'%'
        }
    else:
        return {
            "latest_date": latest_date,
            "latest_price": latest_price,
            "previous_price": prev_price,
            "change": change,
            "change_percentage": str(round(change_percentage,3))+'%'
        }
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api-building:app", host="127.0.0.1", port=8000, reload=True)
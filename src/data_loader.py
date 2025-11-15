import yfinance as yf

def get_stock_info(stock_symbol):
    """
    Retrieve all information about a specific stock.
    
    Args:
        stock_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        dict: Dictionary containing all stock information or None if not found
    """
    try:
        ticker = yf.Ticker(stock_symbol)
        return ticker.info
    except Exception as e:
        print(f"Error retrieving stock information for {stock_symbol}: {e}")
        return None


# Example usage:
stock_data = get_stock_info('AAPL')
print(stock_data)

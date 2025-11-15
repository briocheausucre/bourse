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


def print_stock_parameters(stock_symbol):
    """Print all available stock information parameters in a readable format."""
    stock_data = get_stock_info(stock_symbol)
    
    if stock_data:
        print(f"\nAvailable parameters for {stock_symbol}:\n")
        for i, key in enumerate(sorted(stock_data.keys()), 1):
            print(f"{i:2d}. {key}")
    else:
        print("No data found")


# Example usage:
print_stock_parameters('AAPL')

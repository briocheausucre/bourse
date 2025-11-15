import yfinance as yf
from tabulate import tabulate

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


def main():
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    
    # Get stock data from data_loader module
    stock_data = get_stock_info(ticker)
    
    if stock_data is None:
        print(f"Could not retrieve data for {ticker}")
        return
    
    # Extract 5 most important parameters
    key_metrics = [
        ["Current Price", f"${stock_data.get('currentPrice', 'N/A'):.2f}"],
        ["Market Cap", f"${stock_data.get('marketCap', 'N/A'):,.0f}"],
        ["P/E Ratio", stock_data.get('trailingPE', 'N/A')],
        ["52 Week High", f"${stock_data.get('fiftyTwoWeekHigh', 'N/A'):.2f}"],
        ["Dividend Yield", f"{stock_data.get('dividendYield', 0) * 100:.2f}%"],
    ]
    
    # Display in a nice table
    print(f"\n{'='*40}")
    print(f"Stock Information: {ticker}")
    print(f"{'='*40}\n")
    print(tabulate(key_metrics, headers=["Metric", "Value"], tablefmt="grid"))
    print()

if __name__ == "__main__":
    main()
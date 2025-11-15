from data_loader import get_stock_info
from tabulate import tabulate

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
import yfinance as yf

a = yf.Ticker("AAPL")
keys = a.info.keys()
values = a.info.values()

list = []

for key, value in zip(keys, values):
    if isinstance(value, float):
        #print(f"{key}: {value}")
        list.append(key)
        
print(list)
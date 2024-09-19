import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def MACD_trading_strategy(closing_prices, macd, signal, initial_capital=1000):
    current_capital = initial_capital
    current_shares = 0
    capital_history = [initial_capital]
    max_loss = 0
    max_start = 0
    max_end = 0
    current_start = 0
    current_end = 0
    max_profit =0
    max_profit_start = 0
    max_profit_end = 0

    for i in range(1, len(macd)):
        if macd[i] > signal[i] and macd[i-1] <= signal[i-1] and current_capital > 0:
            shares_to_buy = current_capital / closing_prices[i]
            current_shares += shares_to_buy
            current_capital -= shares_to_buy * closing_prices[i]
            current_start = i
        elif macd[i] < signal[i] and macd[i-1] >= signal[i-1] and current_shares > 0:
            current_capital += current_shares * closing_prices[i]
            current_shares = 0
            current_end = i
            current = (current_capital + current_shares * closing_prices[i]) - capital_history[current_start]
            if current < max_loss:
                max_loss = current
                max_loss_start = current_start
                max_loss_end = current_end
            if current > max_profit:
                max_profit = current
                max_profit_start = current_start
                max_profit_end = current_end
        
        capital_history.append(current_capital + current_shares * closing_prices[i])

    return capital_history, max_loss, max_loss_start, max_loss_end,max_profit,max_profit_start,max_profit_end

def EMA(data,N):
    ema = []
    alpha = 2/(N+1)
    ema_prev = data[0]
    for price in data:
        ema_prev = alpha * price + (1 - alpha) * ema_prev
        ema.append(ema_prev)
    return ema

def MACD(data,N1,N2):
    ema_12 = EMA(data,N1)
    ema_26 = EMA(data,N2)
    macd = [ema_12[i]-ema_26[i] for i in range (len(ema_12))]
    return macd

def SIGNAL(macd,N):
    signal = EMA(macd,N)
    return signal


data = pd.read_csv('nvidia_csv.csv',sep = ',')
date = data['Date']
clossing_prices = data['Close']

macd = MACD(clossing_prices,12,26)
signal = SIGNAL(macd,9)


plt.figure(figsize=(12,6))
plt.plot(clossing_prices,color = 'blue', label = 'Price in $')
plt.legend()
plt.title("NVIDIA STOCK PRICES IN THE LAST 5 YEARS")
plt.show()

histogram = [macd[i] - signal[i] for i in range(len(macd))]
plt.figure(figsize=(12, 6))
plt.bar(range(len(histogram)), histogram, color='green')
plt.axhline(y=0, color='black', linestyle='-')
plt.title('Histogram MACD')
plt.show()

crossings = np.where(np.diff(np.sign(np.array(macd) - np.array(signal))) != 0)[0]
plt.figure(figsize=(12,6))
plt.plot(macd,label = 'MACD',color='blue')
plt.plot(signal,label='SIGNAL',color='red')
plt.scatter(crossings,[signal[i] for i in crossings],color = 'green',label = 'CROSSINGS')
plt.legend()
plt.title("MACD+SIGNAL FOR NVIDIA STOCK PRICES IN THE LAST 5 YEARS")
plt.show()

result, max_loss, max_loss_start, max_loss_end,max_profit,max_profit_start,max_profit_end = MACD_trading_strategy(clossing_prices, macd, signal)
print("Max loss:", max_loss)
print("Start Day:", max_loss_start)
print("End Day:", max_loss_end)
print("Max profit:", max_profit)
print("Start Day:", max_profit_start)
print("End Day:", max_profit_end)
print("End capital:", result[-1])
print(max(result))
print(min(result))
plt.figure(figsize=(18, 6))
plt.plot( result, label='Capital')
plt.title("Capital History Over Time")
plt.xlabel("Days")
plt.ylabel("Capital")
plt.axvline(x=max_loss_start, color='r', linestyle='--', label='Max loss Start')
plt.axvline(x=max_loss_end, color='r', linestyle='--', label='Max loss End')
plt.axvline(x=max_profit_start, color='g', linestyle='--', label='Max profit Start')
plt.axvline(x=max_profit_end, color='g', linestyle='--', label='Max profit End')
plt.legend()
plt.show()

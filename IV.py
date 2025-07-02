import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import scipy.stats as stats 
import math

def black_scholes(s, k, t, r, sigma, option_type):
    
    if t <= 0:
        return 0.0
    if sigma <= 0:
        if option_type == 'call':
            return max(0.0, s - k)
        elif option_type == 'put':
            return max(0.0, k - s)
        else:
            return 0.0 
    
    d1 = (np.log(s / k) + (r + (sigma ** 2) / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type == 'call':
       
        c = s * stats.norm.cdf(d1) - k * math.exp(-r * t) * stats.norm.cdf(d2)
        return c
    elif option_type == 'put': 
        p = k * math.exp(-r * t) * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1)
        return p
    else:
        raise ValueError("Invalid option_type. Must be 'call' or 'put'.")
def vega(s, k, t, r, sigma):
    if t <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(s / k) + (r + (sigma ** 2) / 2) * t) / (sigma * math.sqrt(t))
    vega_value = s * stats.norm.pdf(d1) * math.sqrt(t)
    return vega_value

def implied_volatility(market_price, s, k, t, r, option_type, tolerance = 1e-6, initial_sigma = 0.5, max_iterations = 500):
    if market_price <= 0 or s <= 0 or k <= 0 or t <= 0:
        return np.nan
    if option_type == 'call' and market_price < 1e-6 and s <= k:
        return 0.001 
    if option_type == 'put' and market_price < 1e-6 and s >= k:
        return 0.001 
    
    sigma_n = initial_sigma
    
    for i in range(max_iterations):
        theoretical_price = black_scholes(s, k, t, r, sigma_n, option_type)
        difference = theoretical_price - market_price
        vega_val = vega(s, k, t, r, sigma_n)
        if abs(difference) < tolerance:
            return sigma_n
        if vega_val < 1e-8:
            if abs(difference) < 0.1: 
                return sigma_n
            else: 
                return np.nan
        
        sigma_n = sigma_n - difference / vega_val
        sigma_n = max(0.001, min(sigma_n, 5.0))
    return np.nan


def get_options_data(symbol):
    desired_columns = [
        'strike',
        'lastPrice',
        'option_type'
    ]
    
    def safe_column_selection(df, cols):
        actual_cols = [col for col in cols if col in df.columns]
        return df[actual_cols]
    
    tk = symbol
    expiration_dates = tk.options
    options_data = []
    
    if not expiration_dates:
        print(f"No expiration dates found for {symbol}.")
        return None  
    else:
        for exp_date_str in expiration_dates:
            try:
                option_chain = tk.option_chain(exp_date_str)
                
                calls_df = safe_column_selection(option_chain.calls, desired_columns).copy()
                puts_df = safe_column_selection(option_chain.puts, desired_columns).copy()
                
                current_expiration_dt = pd.to_datetime(exp_date_str).tz_localize('UTC')
                today = pd.Timestamp.now(tz='UTC')
                
                time_to_expiry_seconds = (current_expiration_dt - today).total_seconds()
                time_to_expiry_years = time_to_expiry_seconds / (365 * 24 * 3600)
                
                if time_to_expiry_years < 0:
                    continue
                    
                calls_df['option_type'] = 'call'
                calls_df['expiration_date_str'] = exp_date_str
                calls_df['time_to_expiry'] = time_to_expiry_years
                
                puts_df['option_type'] = 'put'
                puts_df['expiration_date_str'] = exp_date_str
                puts_df['time_to_expiry'] = time_to_expiry_years
                
                combined_exp_df = pd.concat([calls_df, puts_df], ignore_index=True)
                options_data.append(combined_exp_df)
                
            except Exception as e:
                print(f"  Error fetching data for {exp_date_str}: {e}. Skipping this date.")
                continue
    
    if options_data:
        full_options_df = pd.concat(options_data, ignore_index=True)
        full_options_df = full_options_df.drop(columns=['expiration_date_str'])
        full_options_df_sorted = full_options_df.sort_values(
            by=['time_to_expiry', 'option_type', 'strike']
        ).reset_index(drop=True)
        return full_options_df_sorted
    else:
        print(f"No options data was collected for {symbol}.")
        return None

def get_risk_free_rate():
    return 0.04  

def get_current_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        else:
            return None
    except:
        return None

def calculate_implied_volatilities(options_df, current_price, risk_free_rate):
    options_df = options_df.copy()
    
    implied_vols = []
    for _, row in options_df.iterrows():
        iv = implied_volatility(
            market_price=row['lastPrice'],
            s=current_price,
            k=row['strike'],
            t=row['time_to_expiry'],
            r=risk_free_rate,
            option_type=row['option_type']
        )
        implied_vols.append(iv)
    
    options_df['implied_volatility'] = implied_vols
    return options_df


def get_risk_free_rate():
    return 0.04  

def get_current_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        else:
            return None
    except:
        return None

def calculate_implied_volatilities(options_df, current_price, risk_free_rate):
    options_df = options_df.copy()
    
    implied_vols = []
    for _, row in options_df.iterrows():
        iv = implied_volatility(
            market_price=row['lastPrice'],
            s=current_price,
            k=row['strike'],
            t=row['time_to_expiry'],
            r=risk_free_rate,
            option_type=row['option_type']
        )
        implied_vols.append(iv)
    
    options_df['implied_volatility'] = implied_vols
    return options_df

def interactive_options_app():
    print("=== Options Implied Volatility Calculator ===\n")
    
    
    while True:
        symbol = input("Enter stock symbol (e.g., AAPL): ").upper().strip()
        if symbol:
            break
    
    print(f"\nFetching data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    
    current_price = get_current_stock_price(symbol)
    if current_price is None:
        print("Error: Could not fetch current stock price")
        return
    
    print(f"Current stock price: ${current_price:.2f}")
    
    risk_free_rate = get_risk_free_rate()
    print(f"Using fixed risk-free rate: {risk_free_rate:.1%} (4.0%)")
    
    options_df = get_options_data(ticker)
    if options_df is None:
        print("No options data available for this symbol")
        return
    
    print("\nCalculating implied volatilities...")
    options_with_iv = calculate_implied_volatilities(options_df, current_price, risk_free_rate)

    options_with_iv = options_with_iv.dropna(subset=['implied_volatility'])
    
    if options_with_iv.empty:
        print("No valid implied volatilities could be calculated")
        return

    while True:
        print("\n" + "="*60)
        print("OPTIONS SELECTION MENU")
        print("="*60)

        exp_dates = sorted(options_with_iv['time_to_expiry'].unique())
        print("\nAvailable expiration periods (years):")
        for i, exp in enumerate(exp_dates, 1):
            days = int(exp * 365)
            print(f"{i}. {exp:.4f} years ({days} days)")

        try:
            exp_choice = int(input(f"\nSelect expiration (1-{len(exp_dates)}): ")) - 1
            if exp_choice < 0 or exp_choice >= len(exp_dates):
                print("Invalid selection")
                continue
            selected_exp = exp_dates[exp_choice]
        except ValueError:
            print("Please enter a valid number")
            continue

        exp_options = options_with_iv[options_with_iv['time_to_expiry'] == selected_exp].copy()

        print(f"\nOptions for {int(selected_exp * 365)} days to expiration:")
        print("1. Calls")
        print("2. Puts")
        print("3. Both")
        
        try:
            type_choice = int(input("Select option type (1-3): "))
            if type_choice == 1:
                filtered_options = exp_options[exp_options['option_type'] == 'call']
            elif type_choice == 2:
                filtered_options = exp_options[exp_options['option_type'] == 'put']
            elif type_choice == 3:
                filtered_options = exp_options
            else:
                print("Invalid selection")
                continue
        except ValueError:
            print("Please enter a valid number")
            continue
        
        if filtered_options.empty:
            print("No options available for this selection")
            continue

        print(f"\n{'='*80}")
        print(f"IMPLIED VOLATILITY RESULTS FOR {symbol}")
        print(f"Current Price: ${current_price:.2f} | Risk-free Rate: 4.0%")
        print(f"{'='*80}")

        display_df = filtered_options.sort_values(['option_type', 'strike']).copy()

        display_df['Strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
        display_df['Last Price'] = display_df['lastPrice'].apply(lambda x: f"${x:.2f}")
        display_df['Implied Vol'] = display_df['implied_volatility'].apply(lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "N/A")
        display_df['Type'] = display_df['option_type'].str.capitalize()
        display_df['Days to Exp'] = (display_df['time_to_expiry'] * 365).astype(int)

        print(f"{'Type':<6} {'Strike':<10} {'Last Price':<12} {'Implied Vol':<12} {'Days':<6}")
        print("-" * 50)
        
        for _, row in display_df.iterrows():
            print(f"{row['Type']:<6} {row['Strike']:<10} {row['Last Price']:<12} {row['Implied Vol']:<12} {row['Days to Exp']:<6}")

        valid_ivs = filtered_options['implied_volatility'].dropna()
        if not valid_ivs.empty:
            print(f"\nSUMMARY STATISTICS:")
            print(f"Average Implied Volatility: {valid_ivs.mean()*100:.2f}%")
            print(f"Min Implied Volatility: {valid_ivs.min()*100:.2f}%")
            print(f"Max Implied Volatility: {valid_ivs.max()*100:.2f}%")

        print("\nOptions:")
        print("1. Try different expiration/type")
        print("2. Try different stock")
        print("3. Exit")
        
        try:
            next_choice = int(input("What would you like to do? (1-3): "))
            if next_choice == 1:
                continue
            elif next_choice == 2:
                return interactive_options_app()  
            else:
                break
        except ValueError:
            break
    
    print("\nThank you for using the Options Implied Volatility Calculator!")


if __name__ == "__main__":
    interactive_options_app()
import pandas as pd
import numpy as np
from datetime import datetime

class GasStorageCalculator:
    def __init__(self, data_file="Nat_Gas.csv"):
        self.market_data = self.load_market_prices(data_file)
    
    def load_market_prices(self, path):
        df = pd.read_csv(path)
        df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
        df['Prices'] = pd.to_numeric(df['Prices'], errors='coerce')
        df = df.dropna()
        return df.set_index('Dates')
    
    def find_price(self, target_date):
        if target_date in self.market_data.index:
            return float(self.market_data.loc[target_date, 'Prices'])
        
        before_dates = self.market_data[self.market_data.index <= target_date]
        after_dates = self.market_data[self.market_data.index >= target_date]
        
        if len(before_dates) == 0 or len(after_dates) == 0:
            raise ValueError(f"No price data near {target_date}")
        
        date_before = before_dates.index.max()
        date_after = after_dates.index.min()
        
        price_before = float(self.market_data.loc[date_before, 'Prices'])
        price_after = float(self.market_data.loc[date_after, 'Prices'])
        
        total_days = (date_after - date_before).days
        days_to_target = (target_date - date_before).days
        
        if total_days == 0:
            return price_before
        
        calculated_price = price_before + (price_after - price_before) * (days_to_target / total_days)
        return calculated_price
    
    def compute_storage_fee(self, days_stored, capacity, annual_rate):
        return annual_rate * capacity * (days_stored / 365.0)
    
    def evaluate_contract(self,
                        inject_dates,
                        withdraw_dates,
                        buy_prices=None,
                        sell_prices=None,
                        inject_speed=1000,
                        withdraw_speed=1000,
                        max_cap=10000,
                        storage_rate=0.1,
                        trade_volume=None):
        
        if trade_volume is None:
            trade_volume = max_cap
        
        if buy_prices is None:
            buy_prices = [self.find_price(date) for date in inject_dates]
        
        if sell_prices is None:
            sell_prices = [self.find_price(date) for date in withdraw_dates]
        
        total_buy_cost = 0
        total_sell_income = 0
        total_storage_fee = 0
        operation_results = []
        
        for i, (inject_day, withdraw_day, buy_price, sell_price) in enumerate(zip(inject_dates, withdraw_dates, buy_prices, sell_prices)):
            purchase_cost = trade_volume * buy_price
            sales_income = trade_volume * sell_price
            
            storage_days = (withdraw_day - inject_day).days
            storage_fee = self.compute_storage_fee(storage_days, trade_volume, storage_rate)
            
            total_buy_cost += purchase_cost
            total_sell_income += sales_income
            total_storage_fee += storage_fee
            
            net_flow = sales_income - purchase_cost - storage_fee
            unit_profit = sell_price - buy_price - (storage_fee / trade_volume)
            
            operation_results.append({
                'id': i + 1,
                'inject_date': inject_day,
                'withdraw_date': withdraw_day,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'purchase_cost': purchase_cost,
                'sales_income': sales_income,
                'storage_fee': storage_fee,
                'net_flow': net_flow,
                'unit_profit': unit_profit
            })
        
        final_value = total_sell_income - total_buy_cost - total_storage_fee
        avg_unit_profit = final_value / (len(inject_dates) * trade_volume) if len(inject_dates) > 0 else 0
        
        return {
            'final_value': final_value,
            'total_buy_cost': total_buy_cost,
            'total_sell_income': total_sell_income,
            'total_storage_fee': total_storage_fee,
            'avg_unit_profit': avg_unit_profit,
            'operation_count': len(inject_dates),
            'operations': operation_results
        }

def run_examples():
    calculator = GasStorageCalculator("Nat_Gas.csv")
    
    print("Gas Storage Valuation Examples")
    print()
    
    print("Example 1: Seasonal Storage Strategy")
    inject_dates_1 = [
        datetime(2023, 5, 31),
        datetime(2023, 6, 30)
    ]
    withdraw_dates_1 = [
        datetime(2023, 11, 30),
        datetime(2023, 12, 31)
    ]
    
    result1 = calculator.evaluate_contract(
        inject_dates=inject_dates_1,
        withdraw_dates=withdraw_dates_1,
        max_cap=50000,
        trade_volume=10000,
        storage_rate=0.15
    )
    
    print(f"Contract Value: ${result1['final_value']:,.2f}")
    print(f"Per Unit Profit: ${result1['avg_unit_profit']:.2f}")
    print(f"Operations: {result1['operation_count']}")
    print()
    
    print("Example 2: Quick Trading Strategy")
    inject_dates_2 = [
        datetime(2023, 4, 30),
        datetime(2023, 5, 31),
        datetime(2023, 7, 31)
    ]
    withdraw_dates_2 = [
        datetime(2023, 5, 31),
        datetime(2023, 7, 31),
        datetime(2023, 8, 31)
    ]
    
    result2 = calculator.evaluate_contract(
        inject_dates=inject_dates_2,
        withdraw_dates=withdraw_dates_2,
        max_cap=30000,
        trade_volume=5000,
        storage_rate=0.12
    )
    
    print(f"Contract Value: ${result2['final_value']:,.2f}")
    print(f"Per Unit Profit: ${result2['avg_unit_profit']:.2f}")
    print()
    
    print("Example 3: Fixed Price Scenario")
    inject_dates_3 = [datetime(2024, 1, 1)]
    withdraw_dates_3 = [datetime(2024, 3, 1)]
    
    result3 = calculator.evaluate_contract(
        inject_dates=inject_dates_3,
        withdraw_dates=withdraw_dates_3,
        buy_prices=[2.50],
        sell_prices=[4.00],
        max_cap=100000,
        trade_volume=25000,
        storage_rate=0.1
    )
    
    print(f"Contract Value: ${result3['final_value']:,.2f}")
    print(f"Unit Profit: ${result3['operations'][0]['unit_profit']:.2f}")
    print()
    
    print("Detailed Results for Example 1:")
    for op in result1['operations']:
        print(f"Trade {op['id']}:")
        print(f"  Buy: {op['inject_date'].strftime('%Y-%m-%d')} @ ${op['buy_price']:.2f}")
        print(f"  Sell: {op['withdraw_date'].strftime('%Y-%m-%d')} @ ${op['sell_price']:.2f}")
        print(f"  Storage: ${op['storage_fee']:,.2f}")
        print(f"  Net: ${op['net_flow']:,.2f}")
        print(f"  Unit Gain: ${op['unit_profit']:.2f}")
        print()

if __name__ == "__main__":
    run_examples()
import pandas as  pd
import numpy as np 
from typing import Optional, Dict, Any, List
from datetime import datetime,timedelta



class SalesQueryEngine1:
    
    def __init__(self, historical_csv, forecast_csv):
        
        #read historical csv 
        self.historical_df = pd.read_csv(historical_csv)
        self.historical_df['Date'] = pd.to_datetime(self.historical_df['Date'])
        
        #read forecasted csv
        self.forecast_df = pd.read_csv(forecast_csv)
        self.forecast_df['Date'] = pd.to_datetime(self.forecast_df['Date'])
        
        #metadata
        self.stores = sorted(self.historical_df['Store'].unique().tolist())
        self.departments = sorted(self.historical_df['Dept'].unique().tolist())

        print(f"Historical CSV loaded {len(self.historical_df)} , records")
        print(f"Forecasted csv loaded {len(self.forecast_df)}, records")
        print(f"Total number of stores {len(self.stores)}")
        print(f"Total number of departments {len(self.departments)}")
            
    
    def get_combined_df(self, store, dept, start_date = None , end_date = None):
        hist = self.historical_df[
            (self.historical_df['Store']== store) &
            (self.historical_df['Dept'] == dept)
            ].copy()
        
        fut = self.forecast_df[
            (self.forecast_df['Store']== store) &
            (self.forecast_df['Dept']== dept)
        ].copy()

        
        #filtering date and time
        
        #validating and filtering for start date
        if start_date:
            if isinstance(start_date, str):
                start_date = start_date.strip().lower()
                
                if start_date in ['none', 'unknown', 'null'] or '(' in start_date:
                    start_date = None
                    
            if start_date:
                try:
                    start_date = pd.to_datetime(start_date)
                    hist = hist[hist['Date'] >= start_date]
                    fut = fut[fut['Date'] >= start_date]
                except (ValueError, pd.errors.ParserError, TypeError):
                    pass
                
        #validating and filtering for end date
        if end_date:
            if isinstance(end_date, str):
                end_date = end_date.strip().lower()
                
                if end_date in ['none', 'unknown', 'null'] or '(' in end_date:
                    end_date = None
                    
            if end_date:
                try:
                    end_date = pd.to_datetime(end_date)
                    hist = hist[hist['Date'] <= end_date]
                    fut = fut[fut['Date'] <= end_date]
                except(ValueError, pd.errors.ParserError, TypeError):
                    pass
        
        #Dubai dataframe ma sales ko nam uniform rakhya
        hist = hist.rename(columns={'Weekly_Sales': 'Sales'})
        fut = fut.rename(columns={'Predicted_Sales': 'Sales'})
        
        #source vanne column add garne so kunchai fut ra kunchai hist ho thapaos in combined df vanera
        
        hist['Source'] = 'Historical'
        fut['Source'] = 'Forecast'
        
        #combining two dfs
        combined_df = pd.concat([hist,fut]).sort_values('Date').reset_index(drop=True)
        return combined_df

    def total_sales(self,store, dept, start_date = None, end_date = None):
        df=self.get_combined_df(store,dept, start_date, end_date)
        if len(df)==0:
            return 0.0
        else:
            return df['Sales'].sum()
    
    def average_weekly_sales(self, store, dept, start_date= None, end_date= None):
        df = self.get_combined_df(store,dept,start_date,end_date)
        if len(df) == 0:
            return 0.0
        return df['Sales'].mean()
        
    def compare_sales(self, store, dept, year1, year2):
        total1 = self.total_sales(store, dept, f"{year1}-01-01", f"{year1}-12-31")
        total2 = self.total_sales(store, dept, f"{year2}-01-01", f"{year2}-12-31")
        difference = total2-total1
        percentage_changed = ((difference/total1)*100) if total1 !=0 else None
        
        return{
            "year1": year1,
            'year2': year2,
            'year1_total': total1,
            'year2_total': total2,
            'difference': difference,
            '%_changed': percentage_changed
        }
        
    def get_sales_by_month(self,store,dept, year, month):
        year= int(year)
        month= int(month)
        start_date = f"{year}-{month:02d}-01"
        
        #yedi month =12 ho vane jahile 31 days huncha so 
    
        if month == 12:
            end_date = f"{year}-12-31"
            
        #natra nextmonth calculate garne ani  -1 day garne tesari last day of current month tha huncha
        else:
            next_month = datetime(year, month+1,1)
            last_day = next_month - timedelta(days=1)
            end_date = last_day.strftime("%Y-%m-%d")
        
        df = self.get_combined_df(store, dept, start_date, end_date)
        
        if len(df)==0:
            return{
                "total_sales": 0.0,
                "average_weekly_sales": 0.0,
                "num_weeks": 0,
                "date_range": f"{start_date} to {end_date}"
            }
            
        return {
            "total_sales": df['Sales'].sum(),
            "average_weekly_sales": df['Sales'].mean(),
            "num_weeks": len(df),
            "date_range": f"{start_date} to {end_date}"
        }

    def get_sales_trend(self, store, dept, start_date = None, end_date = None, period = 'monthly'):
        #default trend monthly cha but can be weekly or quaterly
        df = self.get_combined_df(store, dept, start_date, end_date)
        
        #if no records find return empty list
        if len(df)==0:
            return []
        
        df = df.copy()
        
        #handling the periods using datetiime dt.to_period
        if period == "monthly":
            df['Periods'] = df['Date'].dt.to_period('M')
        
        elif period == "quarterly":
            df['Periods'] = df['Date'].dt.to_period('Q')
            
        else:
            ##weekly 
            df['Periods'] = df['Date'].dt.to_period('W')
            
        trend = df.groupby('Periods').agg({
            'Sales':['sum', 'mean', 'count']
        }).reset_index()
        
        trend.columns = ['Periods', 'Total_Sales', 'Avg_Sales', 'Num_Records']
        trend['Periods'] = trend['Periods'].astype(str)
        return trend.to_dict('records')

    def get_holiday_impact(self, store, dept, year):
        df = self.get_combined_df(store, dept , start_date= f"{year}-01-01", end_date=f"{year}-12-31")
        
        if len(df) == 0:
            return None
        
        holiday_sales = df[df['IsHoliday']== True]['Sales']
        non_holiday_sales = df[df['IsHoliday']==False]['Sales']
        return {
            "year": year,
            "holiday_total": holiday_sales.sum() if len(holiday_sales) > 0 else 0,
            "holiday_avg": holiday_sales.mean() if len(holiday_sales) > 0 else 0,
            "non_holiday_total": non_holiday_sales.sum() if len(non_holiday_sales) > 0 else 0,
            "non_holiday_avg": non_holiday_sales.mean() if len(non_holiday_sales) > 0 else 0,
            "holiday_weeks": len(holiday_sales),
            "non_holiday_weeks": len(non_holiday_sales),
            "uplift_percentage": ((holiday_sales.mean() / non_holiday_sales.mean() - 1) * 100) 
                                if len(non_holiday_sales) > 0 and non_holiday_sales.mean() > 0 else None
        }

    def get_top_performing_departments(self, store, start_date= None, end_date=None, top_n = 5):
        all_sales = []
        
        for dept in self.departments:
            total = self.total_sales(store, dept, start_date, end_date)
            if total > 0:
                all_sales.append({
                    'department': dept,
                    'total_sales':float(total)
                })
        
        all_sales_df = pd.DataFrame(all_sales)
        
        if len(all_sales_df) == 0:
            return []
        
        top_df = all_sales_df.nlargest(top_n, 'total_sales')
        return top_df.to_dict('records')
    
    def get_sales_summary(self, store, dept, start_date=None, end_date=None):
        
        df = self.get_combined_df(store, dept, start_date, end_date)
        
        if len(df) == 0:
            return {
                "message": "No data available for the specified criteria",
                "store": store,
                "department": dept
            }
        
        historical = df[df['Source'] == 'Historical']
        forecast = df[df['Source'] == 'Forecast']
        
        return {
            "store": store,
            "department": dept,
            "total_sales": df['Sales'].sum(),
            "average_weekly_sales": df['Sales'].mean(),
            "min_weekly_sales": df['Sales'].min(),
            "max_weekly_sales": df['Sales'].max(),
            "std_dev": df['Sales'].std(),
            "total_weeks": len(df),
            "historical_weeks": len(historical),
            "forecast_weeks": len(forecast),
            "date_range": {
                "start": df['Date'].min().strftime("%Y-%m-%d"),
                "end": df['Date'].max().strftime("%Y-%m-%d")
            }
        }

    def search_stores(self, query: str) -> List[str]:
        query_lower = query.lower()
        matches = [s for s in self.stores if query_lower in s.lower()]
        return matches if matches else self.stores

    def search_department(self, query: str) -> List[str]:
        query_lower = query.lower()
        matches = [d for d in self.departments if query_lower in d.lower()]
        return matches if matches else self.departments
    
    
#testing purpose only
if __name__ == "__main__":
    
    historical_path ="data/sales/synthetic_nepal_sales_with_more_holidays.csv"
    forecasted_path = "data/forecasts/all_forecasts_consolidated.csv"
    engine = SalesQueryEngine1(historical_path,forecasted_path)
    gg=engine.get_sales_summary('Bhat-Bhateni', 'Grocery' ,'2026-01-01', '2026-11-11')
    bb= engine.get_combined_df('Kathmandu Mart', 'Grocery', '2022-01-01', '2023-11-11')
    mm = engine.search_department("groce")
    print(mm)
import streamlit as st
import pandas as pd
import numpy_financial as npf
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="DCF Analyzer", layout="wide")

st.title("🚀 DCF Analyzer")
st.subheader("Calculate Intrinsic Value with Discounted Cash Flow Analysis")

class DCFAnalyzer:
    def __init__(self, ticker, beta=1.5, growth_rate_of_operating_expense=0.1, 
                 growth_rate_of_revenue=0.2, gross_margin=0.15, projection_years=5):
        """
        Initialize DCF analyzer with ticker and financial assumptions
        
        Parameters:
        ticker (str): Stock ticker symbol
        beta (float): Stock beta coefficient
        growth_rate_of_operating_expense (float): Projected annual growth rate for operating expenses
        growth_rate_of_revenue (float): Projected annual growth rate for revenue
        gross_margin (float): Projected gross margin percentage
        projection_years (int): Number of years to project future cash flows
        """
        self.ticker = ticker
        self.beta = beta
        self.growth_rate_of_operating_expense = growth_rate_of_operating_expense
        self.growth_rate_of_revenue = growth_rate_of_revenue
        self.gross_margin = gross_margin
        self.projection_years = projection_years
        
        # Initialize market data
        self.risk_free_rate = self._get_risk_free_rate()
        self.market_risk_premium, self.annual_return_sp500 = self._get_market_risk_premium()
        self.cost_of_equity = self.risk_free_rate + (self.beta * self.market_risk_premium)
        
        # Retrieve financial data
        self.historical_data, self.balance_sheet_annual = self._get_financial_data()
        self.dcf = self.historical_data.copy()
        self.dcf.fillna(0, inplace=True)
        self.current_year = int(max(self.dcf.columns[1:]))
        self.number_of_shares = self.balance_sheet_annual.loc[['Ordinary Shares Number']].iloc[0, 0]
    
    def _get_risk_free_rate(self):
        """Fetch current risk-free rate from 10-year Treasury yield"""
        ticker = yf.Ticker("^TNX")
        data = ticker.history(period="5d")
        return data['Close'].iloc[-1] / 100  # Convert to decimal format
    
    def _get_market_risk_premium(self):
        """Calculate market risk premium based on S&P 500 historical returns"""
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(period="10y")
        sp500_data['Return'] = sp500_data['Close'].pct_change()
        
        # Calculate average annual return (assuming daily data)
        annual_return_sp500 = (1 + sp500_data['Return'].mean()) ** 252 - 1  # 252 trading days
        
        # Calculate market risk premium
        market_risk_premium = annual_return_sp500 - self.risk_free_rate
        return market_risk_premium, annual_return_sp500
    
    def _get_financial_data(self):
        """Extract and process financial data for the ticker"""
        ticker = yf.Ticker(self.ticker)
        
        # Fetch financial statements
        quarterly_income = ticker.quarterly_incomestmt
        quarterly_cashflow = ticker.quarterly_cashflow
        quarterly_balance_sheet = ticker.quarterly_balance_sheet
        cash_flow_annual = ticker.cash_flow
        balance_sheet_annual = ticker.balance_sheet
        income_annual = ticker.financials
        
        # Fill NA values
        for i in [quarterly_income, quarterly_cashflow, quarterly_balance_sheet, 
                  cash_flow_annual, balance_sheet_annual, income_annual]:
            i.fillna(0, inplace=True)
        
        # Take only 4 years of data
        cash_flow_annual = cash_flow_annual.iloc[:,0:4]
        balance_sheet_annual = balance_sheet_annual.iloc[:,0:4]
        income_annual = income_annual.iloc[:,0:4]
        
        # Handle missing operating expense
        required_rows = ['Total Revenue', 'Operating Expense', 'Reconciled Depreciation', 'Net Income Common Stockholders']
        if 'Operating Expense' not in income_annual.index:
            income_annual.loc['Operating Expense'] = income_annual.loc['Selling General And Administration'] + income_annual.loc['Other Non Interest Expense']
        
        # Extract income statement rows
        annual_income_filtered = income_annual.loc[['Total Revenue', 'Operating Expense', 'Reconciled Depreciation', 'Net Income Common Stockholders']]
        
        # Check required cash flow rows
        required_rows = ['Operating Cash Flow', 'Capital Expenditure']
        missing_rows = [row for row in required_rows if row not in cash_flow_annual.index]
        if missing_rows:
            raise ValueError(f"The following values are not present in the table: {', '.join(missing_rows)}")
        
        annual_cash_flow_filtered = cash_flow_annual.loc[['Operating Cash Flow', 'Capital Expenditure']]
        
        # Check working capital
        required_rows = ['Working Capital']
        missing_rows = [row for row in required_rows if row not in balance_sheet_annual.index]
        if missing_rows:
            raise ValueError(f"The following values are not present in the table: {', '.join(missing_rows)}")
        
        annual_balance_sheet_filtered = balance_sheet_annual.loc[['Working Capital']]
        
        # Calculate investment in NWC
        nwc_row = balance_sheet_annual.loc[['Working Capital']]
        investment_in_nwc = nwc_row.copy()
        investment_in_nwc.index = ['Investment in NWC']
        investment_in_nwc.iloc[0,:] = nwc_row.iloc[0,:].diff(periods=-1)
        
        # Calculate net capex
        capex_row = cash_flow_annual.loc[['Capital Expenditure']]
        depreciation_row = income_annual.loc[['Reconciled Depreciation']]
        net_capex_row = capex_row.copy()
        net_capex_row.index = ['Net Capex']
        net_capex_row.iloc[0,:] = depreciation_row.iloc[0,:] + capex_row.iloc[0,:]
        
        # Calculate net debt
        debt_row = balance_sheet_annual.loc[['Total Debt']]
        net_debt_row = debt_row.copy()
        net_debt_row.index = ['Net Debt']
        issuance_of_debt_row = cash_flow_annual.loc[['Issuance Of Debt']]
        repayment_of_debt_row = cash_flow_annual.loc[['Repayment Of Debt']]
        net_debt_row.iloc[0,:] = issuance_of_debt_row.iloc[0,:] + repayment_of_debt_row.iloc[0,:]
        
        # Combine all data
        historical_data = pd.concat([
            annual_income_filtered,
            investment_in_nwc,
            annual_balance_sheet_filtered,
            annual_cash_flow_filtered,
            net_capex_row,
            net_debt_row
        ])
        
        # Format dataframe
        historical_data.reset_index(inplace=True)
        historical_data = historical_data.iloc[:,0:5]
        historical_data.rename(columns={'index': 'Breakdown'}, inplace=True)
        
        # Simplify datetime column headers to just the year
        historical_data.columns = ['Breakdown'] + [str(col.year) if isinstance(col, pd.Timestamp) else col for col in historical_data.columns[1:]]
        
        # Sort columns by year
        columns_sorted = ['Breakdown'] + sorted([col for col in historical_data.columns if col != 'Breakdown'])
        historical_data = historical_data[columns_sorted]
        
        return historical_data, balance_sheet_annual
    
    def project_financials(self):
        """Project financial data for future years based on growth assumptions"""
        # Project future revenue
        for i in range(1, self.projection_years + 1):
            self.dcf[str(self.current_year + i)] = 0
            self.dcf.loc[self.dcf['Breakdown'] == 'Total Revenue', str(self.current_year + i)] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Total Revenue', str(self.current_year + i - 1)] * 
                (1 + self.growth_rate_of_revenue)
            )
        
        # Project future operating expenses
        for i in range(1, self.projection_years + 1):
            self.dcf.loc[self.dcf['Breakdown'] == 'Operating Expense', str(self.current_year + i)] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Operating Expense', str(self.current_year + i - 1)] * 
                (1 + self.growth_rate_of_revenue)
            )
        
        # Project future depreciation
        for i in range(1, self.projection_years + 1):
            self.dcf.loc[self.dcf['Breakdown'] == 'Reconciled Depreciation', str(self.current_year + i)] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Reconciled Depreciation', str(self.current_year + i - 1)] * 
                (1 + self.growth_rate_of_revenue)
            )
        
        # Project future working capital
        for i in range(1, self.projection_years + 1):
            self.dcf.loc[self.dcf['Breakdown'] == 'Working Capital', str(self.current_year + i)] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Working Capital', str(self.current_year + i - 1)] * 
                (1 + self.growth_rate_of_revenue)
            )
        
        # Calculate investment in NWC
        for i in range(1, self.projection_years + 1):
            self.dcf.loc[self.dcf['Breakdown'] == 'Investment in NWC', str(self.current_year + i)] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Working Capital', str(self.current_year + i)].values[0] - 
                self.dcf.loc[self.dcf['Breakdown'] == 'Working Capital', str(self.current_year + i - 1)].values[0]
            )
        
        # Project future capital expenditure
        for i in range(1, self.projection_years + 1):
            self.dcf.loc[self.dcf['Breakdown'] == 'Capital Expenditure', str(self.current_year + i)] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Capital Expenditure', str(self.current_year + i - 1)] * 
                (1 + self.growth_rate_of_revenue)
            )
        
        # Calculate net capex
        for i in range(1, self.projection_years + 1):
            rd = self.dcf.loc[self.dcf['Breakdown'] == 'Reconciled Depreciation', str(self.current_year + i)].values[0]
            capex = self.dcf.loc[self.dcf['Breakdown'] == 'Capital Expenditure', str(self.current_year + i)].values[0]
            net_capex = rd + capex
            self.dcf.loc[self.dcf['Breakdown'] == 'Net Capex', str(self.current_year + i)] = net_capex
        
        # Calculate net income
        for i in range(1, self.projection_years + 1):
            self.dcf.loc[self.dcf['Breakdown'] == 'Net Income Common Stockholders', str(self.current_year + i)] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Total Revenue', str(self.current_year + i)].values[0] * 
                (self.gross_margin)
            ) - self.dcf.loc[self.dcf['Breakdown'] == 'Operating Expense', str(self.current_year + i)].values[0] - \
                self.dcf.loc[self.dcf['Breakdown'] == 'Reconciled Depreciation', str(self.current_year + i)].values[0] - \
                self.dcf.loc[self.dcf['Breakdown'] == 'Net Capex', str(self.current_year + i)].values[0]
        
        # Calculate operating cash flow
        for col in self.dcf.columns[1:]:  # Skip the 'Breakdown' column
            self.dcf.loc[self.dcf['Breakdown'] == 'Operating Cash Flow', col] = (
                self.dcf.loc[self.dcf['Breakdown'] == 'Net Income Common Stockholders', col].values[0] +
                self.dcf.loc[self.dcf['Breakdown'] == 'Reconciled Depreciation', col].values[0] +
                self.dcf.loc[self.dcf['Breakdown'] == 'Investment in NWC', col].values[0]
            )
        
        # Project future net debt
        for i in range(1, self.projection_years + 1):
            self.dcf.loc[self.dcf['Breakdown'] == 'Net Debt', str(self.current_year + i)] = \
                self.dcf.loc[self.dcf['Breakdown'] == 'Net Debt', str(self.current_year)]
        
        # Calculate FCFE (Free Cash Flow to Equity)
        fcfe_row = ['FCFE']
        for i in range(len(self.dcf.columns[1:])):
            fcfe_row.append(
                self.dcf.loc[self.dcf['Breakdown'] == 'Operating Cash Flow'].iloc[0, 1:][i] +
                self.dcf.loc[self.dcf['Breakdown'] == 'Net Capex'].iloc[0, 1:][i] +
                self.dcf.loc[self.dcf['Breakdown'] == 'Net Debt'].iloc[0, 1:][i]
            )
        
        self.dcf.loc[len(self.dcf)] = fcfe_row
        return self.dcf
    
    def calculate_npv(self):
        """Calculate the Net Present Value of projected cash flows"""
        cashflows = []
        for i in range(1, self.projection_years + 1):
            cashflows.append(self.dcf.loc[self.dcf['Breakdown'] == 'FCFE', str(self.current_year + i)].values[0])
        
        npv = npf.npv(self.cost_of_equity, cashflows)
        return npv
    
    def calculate_intrinsic_price_per_share(self):
        """Calculate intrinsic price per share based on NPV"""
        npv = self.calculate_npv()
        ipp = npv / self.number_of_shares
        return ipp
    
    def run_dcf_analysis(self):
        """Run complete DCF analysis and return results"""
        self.project_financials()
        npv = self.calculate_npv()
        ipp = self.calculate_intrinsic_price_per_share()
        
        return {
            "dcf_data": self.dcf,
            "npv": npv,
            "intrinsic_price": ipp,
            "cost_of_equity": self.cost_of_equity,
            "risk_free_rate": self.risk_free_rate,
            "market_risk_premium": self.market_risk_premium,
            "current_year": self.current_year,
            "projection_years": self.projection_years
        }
    
    def get_summary(self):
        """Get a summary of the DCF analysis results"""
        results = self.run_dcf_analysis()
        
        summary = {
            "ticker": self.ticker,
            "intrinsic_price": results["intrinsic_price"],
            "npv": results["npv"],
            "cost_of_equity": results["cost_of_equity"],
            "risk_free_rate": results["risk_free_rate"],
            "market_risk_premium": results["market_risk_premium"],
            "beta": self.beta,
            "growth_rate_of_revenue": self.growth_rate_of_revenue,
            "gross_margin": self.gross_margin,
            "projection_years": self.projection_years
        }
        
        return summary

# User Input Section
with st.sidebar:
    st.header("📊 Model Parameters")
    
    # Company ticker input
    ticker = st.text_input("Stock Ticker Symbol", value="TSLA", 
                           help="Enter the stock ticker symbol (e.g., AAPL, MSFT, TSLA)")
    
    st.subheader("Financial Assumptions")
    
    # Beta input
    beta = st.number_input("Beta Coefficient (Default: 1.5)", 
                           min_value=0.1, max_value=3.0, value=1.5, step=0.1,
                           help="Stock's volatility relative to the market")
    
    # Growth rate inputs
    growth_rate_of_operating_expense = st.number_input("Operating Expense Growth Rate (Default: 0.1)", 
                                                       min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                                                       format="%.2f",
                                                       help="Projected annual growth rate for operating expenses")
    
    growth_rate_of_revenue = st.number_input("Revenue Growth Rate (Default: 0.2)", 
                                             min_value=0.01, max_value=0.5, value=0.2, step=0.01,
                                             format="%.2f",
                                             help="Projected annual growth rate for revenue")
    
    # Gross margin input
    gross_margin = st.number_input("Gross Margin (Default: 0.15)", 
                                   min_value=0.01, max_value=1.0, value=0.15, step=0.01,
                                   format="%.2f",
                                   help="Projected gross margin percentage")
    
    # Projection years input
    projection_years = st.number_input("Projection Years (Default: 5)", 
                                       min_value=1, max_value=10, value=5, step=1,
                                       help="Number of years to project future cash flows")
    
    # Run analysis button
    run_analysis = st.button("Run DCF Analysis")

# Function to run the analysis
def perform_analysis():
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            # Create DCF analyzer with user inputs
            dcf = DCFAnalyzer(
                ticker=ticker,
                beta=beta,
                growth_rate_of_operating_expense=growth_rate_of_operating_expense,
                growth_rate_of_revenue=growth_rate_of_revenue,
                gross_margin=gross_margin,
                projection_years=projection_years
            )
            
            # Run the analysis
            results = dcf.run_dcf_analysis()
            
            # Display results
            st.success(f"Analysis completed for {ticker}")
            
            # Display metrics
            st.header("Key Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Intrinsic Price Per Share", f"${results['intrinsic_price']:.2f}")
                st.metric("Risk-Free Rate", f"{results['risk_free_rate']:.2%}")
            
            with metrics_col2:
                # Convert NPV to millions
                npv_in_millions = results['npv'] / 1_000_000
                st.metric("Net Present Value", f"${npv_in_millions:.2f}M")
                st.metric("Market Risk Premium", f"{results['market_risk_premium']:.2%}")
                
            with metrics_col3:
                st.metric("Cost of Equity", f"{results['cost_of_equity']:.2%}")
                st.metric("Projection Years", f"{results['projection_years']}")
            
            # DCF Table
            st.header("DCF Analysis Table")
            
            # Convert numerical columns to millions for display
            display_df = results['dcf_data'].copy()
            
            # Convert all numeric columns to millions (except the first column which is 'Breakdown')
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col] / 1_000_000
            
            # Style the dataframe
            def format_millions(x):
                if isinstance(x, (int, float)):
                    return f"${x:.2f}"
                return x
            
            # Apply formatting and display
            st.dataframe(display_df.style.format(format_millions, subset=display_df.columns[1:]), 
                         use_container_width=True)
            
            # Add a note explaining the values
            st.caption("Note: All financial values are in millions of dollars.")
            
            # Download button for the DCF table
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download DCF Table as CSV",
                data=csv,
                file_name=f"{ticker}_DCF_Analysis.csv",
                mime="text/csv",
            )
            
            # Chart for visualizing FCFE
            st.header("Free Cash Flow to Equity (FCFE) Projection")
            
            # Extract FCFE data for plotting (already in millions)
            fcfe_data = display_df[display_df['Breakdown'] == 'FCFE'].iloc[:, 1:].values.flatten()
            years = display_df.columns[1:]
            
            # Create a DataFrame for plotting
            fcfe_df = pd.DataFrame({
                'Year': years,
                'FCFE (in millions)': fcfe_data
            })
            
            # Plot the FCFE
            st.bar_chart(fcfe_df.set_index('Year'))
            
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            st.info("Please check the ticker symbol and try again.")

# Main content
if run_analysis or 'run_clicked' in st.session_state:
    st.session_state['run_clicked'] = True
    perform_analysis()
else:
    st.info("👈 Enter your parameters in the sidebar and click 'Run DCF Analysis' to start.")
    st.markdown("""
    ### Welcome to the DCF Analyzer
    
    This tool helps you perform a Discounted Cash Flow (DCF) analysis for any publicly traded company. 
    
    **How to use:**
    1. Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA)
    2. Adjust the financial assumptions as needed
    3. Click "Run DCF Analysis" to see the results
    
    **What you'll get:**
    - Intrinsic price per share
    - Net present value (NPV) of future cash flows
    - Detailed DCF projection table (all values in millions)
    - Visualization of free cash flows
    
    All calculations are based on real financial data from Yahoo Finance and your specified growth assumptions.
    """)
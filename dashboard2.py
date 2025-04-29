import streamlit as st 
import plotly.express as px 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os 
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima, ARIMA, model_selection


warnings.filterwarnings('ignore') 

def filtering(df, store_number, product_family):
    filtered = df[
    (df['store_nbr'] == store_number) &
    (df['family'] == product_family)].copy()
    new_df = filtered[['date', 'sales', 'onpromotion', 'is_holiday']].copy()
    new_df = new_df.set_index('date')
    return new_df

st.set_page_config(page_title='Sales Forecast!!', page_icon=":bar_chart:", layout='wide')

st.title(" 📈 Sales Forecasting Dashboard")

fl = st.file_uploader(":file_folder: Upload a file", type=['csv', 'txt', 'xlsx', 'xls'])

if fl is None:
    st.write('Please upload the file named (final_forecast.csv)')
elif fl.name != 'final_forecast.csv':
    st.write(f'You uploaded "{fl.name}". Please upload the correct file named "final_forecast.csv"')
else:
    filename = fl.name
    st.write(f"Uploaded file: {filename}")
    main_df = pd.read_csv(filename, encoding = 'ISO-8859-1')

    main_df['date'] = pd.to_datetime(main_df['date'])

    split_date = pd.to_datetime('2017/08/16')
    df = main_df[main_df["date"] < split_date].copy()
    test_df = main_df[main_df["date"]>= split_date].copy()

    tab1, tab2 = st.tabs(["Analysis", "Forecasting"])

    with tab1: 
        st.header('Start Analyzing !! 📈')

        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)

        start_date = pd.to_datetime(df['date'], dayfirst=True).min()
        end_date = pd.to_datetime(df['date'], dayfirst=True).max()

        with col1:
            date1 = pd.to_datetime(st.date_input('Start date', start_date))

        with col2:
            date2 = pd.to_datetime(st.date_input('End date', end_date))

        df = df[(df['date'] >= date1) & (df['date'] <= date2)].copy()
    # sidebar 
       
        st.sidebar.header('Choose your filter:')

        # Create cascading filters
        # 1. State filter
        sel_state = st.sidebar.multiselect('Pick your state', df['state'].unique())
        if not sel_state:
            filtered_df = df.copy()
        else:
            filtered_df = df[df['state'].isin(sel_state)]

        # 2. City filter (dependent on state selection)
        sel_city = st.sidebar.multiselect('Pick your city', filtered_df['city'].unique())
        if sel_city:
            filtered_df = filtered_df[filtered_df['city'].isin(sel_city)]

        # 3. Store filter (dependent on previous selections)
        sel_store = st.sidebar.multiselect('Pick your store', filtered_df['store_nbr'].unique())
        if sel_store:
            filtered_df = filtered_df[filtered_df['store_nbr'].isin(sel_store)]


        category_df = filtered_df.groupby(by = ['family'], as_index = False)['sales'].sum()

        # category bar chart
        st.subheader('Sales per Category')
        fig = px.bar(category_df, x = 'family', y = 'sales', text=['${:,.2f}'.format(x) for x in category_df['sales']],
        template = 'seaborn')
        st.plotly_chart(fig, use_container_width=True, height=400)
        # table and download
        with st.expander('Category view data'):
            st.write(category_df.style.background_gradient(cmap='Blues'))
            csv = category_df.to_csv(index = False).encode('utf-8')
            st.download_button('Download Data', data = csv, file_name = 'Category.csv', mime = 'text/csv',
                                help = 'Click here to download the data as CSV file')

        
        st.subheader('Sales per State')
        fig = px.pie(filtered_df, values = 'sales', names = 'state', hole = 0.5)
        fig.update_traces(text = filtered_df['state'], textposition = 'outside')
        st.plotly_chart(fig, use_container_width=True) 
        # table and download 
        with st.expander('States View Data'):
            state = filtered_df.groupby(by = 'state', as_index = False)['sales'].sum()
            st.write(state.style.background_gradient(cmap='Oranges'))
            csv = state.to_csv(index = False).encode('utf-8')
            st.download_button('Download Data', data = csv, file_name = 'State.csv', mime = 'text/csv',
                                help = 'Click here to download the data as CSV file')

        # create hierarchichal tree
        st.subheader('Hierarchical view of Sales using TreeMap')
        fig3 = px.treemap(filtered_df, path = ['state', 'city', 'store_nbr', 'family'], values = 'sales', hover_data = ['sales'],
                    color = 'family')
        fig3.update_layout(width=800, height= 800)
        st.plotly_chart(fig3, use_container_width= True)

        # barchart for category sales
        st.subheader('Category Sales')
        categories = filtered_df['family'].unique()

        selected_category = st.selectbox(
            label='Select a Category',
            options= categories
        )

        category_data = filtered_df[filtered_df['family'] == selected_category]
        sales_by_store = category_data.groupby('store_nbr')['sales'].sum().reset_index()

        sales_by_store = sales_by_store.sort_values('sales', ascending=True)

        # Plotly horizontal bar chart
        fig = px.bar(
            sales_by_store,
            x='sales',
            y='store_nbr',
            orientation='h',
            title=f'Sales by Store for {selected_category}',
            labels={'sales': 'Total Sales', 'store_nbr': 'Store Number'},
            height=600,  # adjust this value as needed
        )
        st.plotly_chart(fig)

        with st.expander('Store sales per Category'):
            st.write(sales_by_store.style.background_gradient(cmap='Blues'))
            csv = sales_by_store.to_csv(index = False).encode('utf-8')
            st.download_button('Download Data', data= csv, file_name= 'storesalespercategory.csv', mime = 'text/csv',
                                help = 'Click here to download the data as CSV')

    # TIME SERIES
    with tab2: 
        # Make sure your DataFrame is loaded (using your variable name `mdf`)

        st.header('Start forecasting 📈!!')

        
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df['month'] = df['date'].dt.month_name()

        st.subheader("Monthly Sales by Category")

        # Get unique categories
        categories = df['family'].unique()
        years = df['date'].dt.year.unique()

        # Category selection
        col3, col4 = st.columns(2)

        with col3: 
            selected_category = st.selectbox("Select Category", options=categories)

        with col4:
            selected_year = st.selectbox('Select year', options=years)

        df = df[df['date'].dt.year == selected_year]

        # Filter and group data
        category_df = df[df['family'] == selected_category]
        monthly_sales = category_df.groupby('month')['sales'].sum().reset_index()

        # Optional: Ensure months are in correct order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
        monthly_sales['month'] = pd.Categorical(monthly_sales['month'], categories=month_order, ordered=True)
        monthly_sales = monthly_sales.sort_values('month')

        # Plot the line chart
        fig = px.line(monthly_sales, x='month', y='sales', title=f"Monthly Sales for '{selected_category}'", markers=True)

        fig.update_layout( xaxis_title="Month", yaxis_title="Total Sales", height=500)

        st.plotly_chart(fig)
        
        with st.expander('Tabular data'): # download the results from graph
            st.write(monthly_sales.style.background_gradient(cmap='Blues'))
            csv = monthly_sales.to_csv(index = False).encode('utf-8')
            st.download_button('Download Data', data = csv, file_name = 'monthly.csv', mime = 'text/csv',
                                help = 'Click here to download the data as CSV file')

    # Time series. Seasonal Decomposition
        
        st.subheader('Seasonality and Trend')
        stores = df['store_nbr'].unique()

        col5, col6 = st.columns(2)

        with col5: 
            selected_store = st.selectbox('Select store', options=stores)
        with col6:
            selected_category = st.selectbox("Select Category", options=categories, key = 'no2')

        ts_df = filtering(df, selected_store, selected_category)

        weekly_sales = ts_df['sales'].resample('W').mean()
        decomposition = seasonal_decompose(weekly_sales, period=7)  
        # Plot the decomposition
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Observed', 'Trend', 'Seasonal'))

        # Add traces
        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed', line=dict(color='greenyellow')), row=1, col=1)

        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend',line=dict(color='aliceblue')), row=2, col=1)

        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal', line=dict(color='red')), row=3, col=1)

        # Update layout
        fig.update_layout(height=800, title_text=f"Seasonal Decomposition for Store {selected_store}, {selected_category}", showlegend=False)

        # Update y-axis titles
        fig.update_yaxes(title_text="Observed", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)


#####################################################################

        st.subheader('Forecasting')

    
        asd = filtering(main_df, selected_store, selected_category) # main_df
        new_dates = pd.date_range('2017-09-01', '2017-09-14')
        # creating 15 more days for september
        new_rows = pd.DataFrame({
            'sales': np.nan,
            'onpromotion': 0,  # Default value - change as needed
            'is_holiday': 0    # Default value - change as needed
        }, index=new_dates)

        # Combine with existing data
        extended_df = pd.concat([asd, new_rows]).sort_index()

        ref_dates = new_dates - pd.DateOffset(years=1)

        # Copy promotion/holiday data from 2016
        for new_date, ref_date in zip(new_dates, ref_dates):
            if ref_date in asd.index:
                extended_df.loc[new_date, 'onpromotion'] = asd.loc[ref_date, 'onpromotion']
                extended_df.loc[new_date, 'is_holiday'] = asd.loc[ref_date, 'is_holiday']

        split_date = pd.to_datetime('2017/08/16')
        df_test = extended_df[extended_df.index >= split_date].copy()
        df_train = extended_df[extended_df.index < split_date].copy()

        if st.button('Forecast'):
            factor = df_train[['sales']] # train data
            ex = df_train.drop(columns=['sales']) # train external factor

            model_sarimax = auto_arima(factor, m =7, X = ex)
            model_sarimax.summary()

            # forecast
            exg_test = df_test.drop(columns=['sales']) 
            pred = model_sarimax.predict(n_periods=30, X = exg_test)

            print("Predictions:", pred)

            # confidence interval
            
            confidence_percentage = 0.4  # 40% bands
            lower_bound = pred * (1 - confidence_percentage)
            upper_bound = pred * (1 + confidence_percentage)

            # Create Plotly figure
            fig = go.Figure()

            # Add traces
            fig.add_trace(go.Scatter(x=df_train.index, y=df_train['sales'], name='Actual Data', line=dict(color='blue')))

            fig.add_trace(go.Scatter(x=exg_test.index, y=pred, name='Prediction', line=dict(color='red')))

            # Add confidence interval
            fig.add_trace(go.Scatter(x=exg_test.index, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False))

            fig.add_trace(go.Scatter( x=exg_test.index, y=lower_bound, mode='lines',line=dict(width=0), fillcolor='rgba(0, 100, 80, 0.2)', fill='tonexty', name='Confidence Range'))

            # Update layout
            fig.update_layout(title=f'Train and Forecast with {selected_category}', xaxis_title='Date', yaxis_title='Sales', hovermode='x unified', showlegend=True, template='plotly_white', plot_bgcolor='white', height=600)

            # Add grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Show in Streamlit
            st.plotly_chart(fig)

                # for table   

            pred_df = pd.DataFrame(
                data=np.array(pred).reshape(-1, 1),  # Explicitly reshape as needed
                index=exg_test.index,
                columns=[f"Store {selected_store}, {selected_category} Forecast"]
                )
            # pred_df.index = pred_df.index.date  # Removes time component
            # pred_df.index.name = 'Date'
            with st.expander('Forecast Results', expanded=True):
                st.write(pred_df.style.background_gradient(cmap='Oranges'))
                csv = pred_df.to_csv().encode('utf-8')
                st.download_button('Download Data', data = csv, file_name = f'forecast_store_{selected_store}_{selected_category}.csv', mime = 'text/csv',
                                help = 'Click here to download the data as CSV file')
        else:
            st.write('Click to Forecast')


        
    
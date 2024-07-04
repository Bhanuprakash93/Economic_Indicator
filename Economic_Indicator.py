# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from pycaret.time_series import TSForecastingExperiment
# # from sklearn.metrics import mean_absolute_percentage_error
# # from matplotlib.dates import DateFormatter



# # # Function to load data
# # @st.cache_data
# # def load_data(file_path):
# #     data = pd.read_excel(file_path)
# #     data.set_index('observation_date', inplace=True)
# #     return data

# # # Function to create and plot model predictions
# # def plot_predictions(train, fh, model_code='auto_arima'):
# #     exp = TSForecastingExperiment()
# #     exp.setup(data=train, fh=fh, fold=1, session_id=42)
    
# #     model = exp.create_model(model_code)
# #     final_model = exp.finalize_model(model)
# #     y_predict = exp.predict_model(final_model)
    
# #     return y_predict, final_model, exp

# # # Function to plot full data
# # def plot_full_data(data):
# #     fig, ax = plt.subplots(figsize=(10, 6))
    
# #     # Plot full data
# #     ax.plot(data.index, data['MRTSSM4451USS'], marker='o', linestyle='-', color='b', label='Actual')

# #     ax.set_xlabel('Time')
# #     ax.set_ylabel('Value')
# #     ax.set_title('Actual Values (Full Data)')
# #     ax.legend()
# #     ax.grid(True)
    
# #     # Format x-axis to show month and year
# #     ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

# #     return fig

# # # Function to plot predicted data
# # def plot_predicted_data(y_predict, fh):
# #     fig, ax = plt.subplots(figsize=(10, 6))
    
# #     # Convert PeriodIndex to DatetimeIndex
# #     y_predict.index = y_predict.index.to_timestamp()
    
# #     # Plot predicted data
# #     y_pred_index = pd.date_range(start=y_predict.index[0], periods=fh, freq='MS')
# #     ax.plot(y_pred_index, y_predict['y_pred'], marker='o', linestyle='--', color='r', label='Predicted')

# #     ax.set_xlabel('Time')
# #     ax.set_ylabel('Value')
# #     ax.set_title(f'Predicted Values for {fh} Months')
# #     ax.legend()
# #     ax.grid(True)
    
# #     # Format x-axis to show month and year
# #     ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

# #     return fig

# # # Streamlit app
# # st.set_page_config(layout='wide', page_title='Economic Indicator(FRED Data)', page_icon=":chart_with_upwards_trend:")



# # # Page title
# # st.title('Economic Indicator Forecast (FRED Data)')

# # # File uploader
# # file_path = st.text_input('Enter the path to your Excel file:', 'C:/Users/Bhanu prakash/OneDrive - Vijaybhoomi International School/Desktop/Pycaret_SS/MRTSSM4451USS_edited.xlsx')

# # if file_path:
# #     try:
# #         data = load_data(file_path)

# #         # Display the first few rows of the dataframe and start/end dates
# #         st.write('Data preview:')
# #         st.write(data.head())
# #         st.write(f"Start Date: {data.index.min().strftime('%b %Y')}")
# #         st.write(f"End Date: {data.index.max().strftime('%b %Y')}")

# #         # Select the training and testing split
# #         st.sidebar.subheader('Select Tab')
# #         tab_selection = st.sidebar.radio('Select your tab:', ('Train Test split, validating data', 'Full data predictions'))

# #         if tab_selection == 'Train Test split, validating data':
# #             st.subheader('Train Test split, validating data')

# #             st.sidebar.subheader('Select Training and Testing Split')
# #             start_date = st.sidebar.date_input('Training start date', value=pd.to_datetime('1992-01-01'), min_value=pd.to_datetime('1992-01-01'))
# #             end_date = st.sidebar.date_input('Training end date', value=pd.to_datetime('2023-03-01'))
            
# #             # Convert start_date and end_date to datetime
# #             start_date = pd.to_datetime(start_date)
# #             end_date = pd.to_datetime(end_date)
            
# #             train = data[(data.index >= start_date) & (data.index <= end_date)]
# #             test = data[data.index > end_date]

# #             max_fh = len(test)
# #             st.write(f'Maximum forecast horizon (fh) based on test data length: {max_fh} months')
            
# #             # Select the number of months to predict
# #             fh = st.slider('Number of months to predict:', min_value=1, max_value=max_fh, value=12)

# #             # Enter button to generate predictions
# #             if st.button('Enter'):
# #                 # Plot the predictions for the selected forecast horizon
# #                 st.subheader(f'Predictions for {fh} Months')
# #                 y_predict, final_model, exp = plot_predictions(train, fh)

# #                 actual_values = test['MRTSSM4451USS'][:fh].values
# #                 predicted_values = y_predict['y_pred'].values

# #                 # Plot actual vs predicted values for the selected forecast horizon
# #                 fig, ax = plt.subplots(figsize=(10, 6))
# #                 time_index = pd.date_range(start=end_date + pd.DateOffset(months=1), periods=fh, freq='MS')
# #                 ax.plot(time_index, actual_values, marker='o', linestyle='-', color='b', label='Actual')
# #                 ax.plot(time_index, predicted_values, marker='o', linestyle='--', color='r', label='Predicted')
# #                 ax.set_xlabel('Time')
# #                 ax.set_ylabel('Value')
# #                 ax.set_title(f'Actual vs Predicted Values (Test Data for {fh} Months)')
# #                 ax.legend()
# #                 ax.grid(True)
                
# #                 # Format x-axis to show month and year
# #                 ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
                
# #                 # Display the plot using st.pyplot()
# #                 st.pyplot(fig)

# #                 # Calculate and display MAPE for the selected forecast horizon
# #                 mape = mean_absolute_percentage_error(actual_values, predicted_values)
# #                 st.markdown(f'**Mean Absolute Percentage Error (MAPE) for {fh} months: {mape}**')

# #         elif tab_selection == 'Full data predictions':
# #             st.subheader('Full data predictions')

# #             # Select the number of months to predict for full data
# #             fh_full = st.slider('Number of months to predict for full data:', min_value=1, max_value=12, value=12)

# #             # Enter button to generate predictions
# #             if st.button('Enter for Full Data Predictions'):
# #                 # Initialize TSForecastingExperiment for full data predictions
# #                 exp = TSForecastingExperiment()
# #                 exp.setup(data=data, fh=fh_full, fold=1, session_id=42)
# #                 best_model = exp.create_model('auto_arima')
# #                 final_model = exp.finalize_model(best_model)
# #                 y_predict_fulldata = exp.predict_model(final_model)

# #                 # Plot the full data first
# #                 st.subheader('Full Data Plot')
# #                 fig_full = plot_full_data(data)
# #                 st.pyplot(fig_full)

# #                 # Plot the predicted values for full data predictions
# #                 st.subheader(f'Predicted Values for Full Data ({fh_full} Months)')
# #                 fig_predicted = plot_predicted_data(y_predict_fulldata, fh_full)
# #                 st.pyplot(fig_predicted)

# #                 # Display the predicted values for full data predictions
# #                 st.write('Predicted Values:')
# #                 st.write(y_predict_fulldata)

# #     except FileNotFoundError:
# #         st.error("The specified file path is not valid. Please enter a valid file path.")
# # else:
# #     st.info("Please enter a file path to load the data.")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from pycaret.time_series import TSForecastingExperiment
# from sklearn.metrics import mean_absolute_percentage_error
# from matplotlib.dates import DateFormatter
# from matplotlib.ticker import FuncFormatter

# # Set page config
# st.set_page_config(layout='wide', page_title='Economic Indicator (FRED Data)', page_icon=":chart_with_upwards_trend:")

# # Access secrets from secrets.toml
# file_path = st.secrets["data"]["file_path"]

# # Function to load data
# @st.cache_data
# def load_data(file_path):
#     data = pd.read_excel(file_path)
#     data.set_index('observation_date', inplace=True)
#     return data

# # Function to create and plot model predictions
# def plot_predictions(train, fh, model_code='auto_arima'):
#     exp = TSForecastingExperiment()
#     exp.setup(data=train, fh=fh, fold=1, session_id=42)
    
#     model = exp.create_model(model_code)
#     final_model = exp.finalize_model(model)
#     y_predict = exp.predict_model(final_model)
    
#     return y_predict, final_model, exp

# # Function to plot full data
# def plot_full_data(data):
#     fig, ax = plt.subplots(figsize=(16, 5))
    
#     # Plot full data
#     ax.plot(data.index, data['MRTSSM4451USS'], marker='o', linestyle='-', color='b', label='Actual')

#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.set_title('Actual Values (Full Data)')
#     ax.legend()
#     ax.grid(True)
    
#     # Format x-axis to show month and year
#     ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

#     # Format y-axis to include commas for thousands separators
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

#     return fig

# # Function to plot predicted data
# def plot_predicted_data(y_predict, fh):
#     fig, ax = plt.subplots(figsize=(16, 5))
    
#     # Convert PeriodIndex to DatetimeIndex
#     y_predict.index = y_predict.index.to_timestamp()
    
#     # Plot predicted data
#     y_pred_index = pd.date_range(start=y_predict.index[0], periods=fh, freq='MS')
#     ax.plot(y_pred_index, y_predict['y_pred'], marker='o', linestyle='--', color='r', label='Predicted')

#     # Set x-axis ticks to monthly intervals
#     ax.set_xticks(y_pred_index)
    
#     # Format x-axis to show month and year
#     ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.set_title(f'Predicted Values for {fh} Months')
#     ax.legend()
#     ax.grid(True)

#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)

#     # Format y-axis to include commas for thousands separators
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
#     return fig

# # Function to plot actual vs predicted values
# def plot_actual_vs_predicted(time_index, actual_values, predicted_values, fh):
#     fig, ax = plt.subplots(figsize=(16, 6))
    
#     # Plot actual vs predicted values for the selected forecast horizon
#     ax.plot(time_index, actual_values, marker='o', linestyle='-', color='b', label='Actual')
#     ax.plot(time_index, predicted_values, marker='o', linestyle='--', color='r', label='Predicted')
    
#     # Set x-axis ticks to monthly intervals
#     ax.set_xticks(time_index)
    
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.set_title(f'Actual vs Predicted Values (Test Data for {fh} Months)')
#     ax.legend()
#     ax.grid(True)
    
#     # Format x-axis to show month and year
#     ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)

#     # Format y-axis to include commas for thousands separators
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
#     return fig

# # Page title
# st.title('Economic Indicator Forecast (FRED Data)')

# try:
#     data = load_data(file_path)

#     # Display the first few rows of the dataframe and start/end dates
#     st.write('Data preview:')
#     st.write(data.head())
#     st.write(f"Start Date: {data.index.min().strftime('%b %Y')}")
#     st.write(f"End Date: {data.index.max().strftime('%b %Y')}")

#     # Select the training and testing split
#     st.sidebar.subheader('Select Tab')
#     tab_selection = st.sidebar.radio('Select your tab:', ('Train Test split, validating data', 'Full data predictions'))

#     if tab_selection == 'Train Test split, validating data':
#         st.subheader('Train Test split, validating data')

#         st.sidebar.subheader('Select Training and Testing Split')
#         start_date = st.sidebar.date_input('Training start date', value=pd.to_datetime('1992-01-01'), min_value=pd.to_datetime('1992-01-01'))
#         end_date = st.sidebar.date_input('Training end date', value=pd.to_datetime('2023-03-01'))
        
#         # Convert start_date and end_date to datetime
#         start_date = pd.to_datetime(start_date)
#         end_date = pd.to_datetime(end_date)
        
#         train = data[(data.index >= start_date) & (data.index <= end_date)]
#         test = data[data.index > end_date]

#         max_fh = len(test)
#         st.write(f'Maximum forecast horizon (fh) based on test data length: {max_fh} months')
        
#         # Select the number of months to predict
#         fh = st.slider('Number of months to predict:', min_value=1, max_value=max_fh, value=12)

#         # Enter button to generate predictions
#         if st.button('Enter'):
#             # Plot the predictions for the selected forecast horizon
#             st.subheader(f'Predictions for {fh} Months')
#             y_predict, final_model, exp = plot_predictions(train, fh)

#             actual_values = test['MRTSSM4451USS'][:fh].values
#             predicted_values = y_predict['y_pred'].values

#             # Round off the predicted values
#             predicted_values = predicted_values.round()

#             # Plot actual vs predicted values for the selected forecast horizon
#             time_index = pd.date_range(start=end_date + pd.DateOffset(months=1), periods=fh, freq='MS')
#             fig = plot_actual_vs_predicted(time_index, actual_values, predicted_values, fh)
            
#             # Display the plot using st.pyplot()
#             st.pyplot(fig)

#             # Calculate and display MAPE for the selected forecast horizon
#             mape = mean_absolute_percentage_error(actual_values, predicted_values)
#             st.markdown(f'**Mean Absolute Percentage Error (MAPE) for {fh} months: {mape}**')

#             # Display the predicted values with month and year only
#             predicted_df = pd.DataFrame({
#                 'Date': time_index.strftime('%b %Y'),
#                 'Predicted Value': predicted_values
#             })
#             st.write('Predicted Values:')
#             st.write(predicted_df)

#     elif tab_selection == 'Full data predictions':
#         st.subheader('Full data predictions')

#         # Select the number of months to predict for full data
#         fh_full = st.slider('Number of months to predict for full data:', min_value=1, max_value=12, value=12)

#         # Enter button to generate predictions
#         if st.button('Enter for Full Data Predictions'):
#             # Initialize TSForecastingExperiment for full data predictions
#             exp = TSForecastingExperiment()
#             exp.setup(data=data, fh=fh_full, fold=1, session_id=42)
#             best_model = exp.create_model('auto_arima')
#             final_model = exp.finalize_model(best_model)
#             y_predict_fulldata = exp.predict_model(final_model)

#             # Round off the predicted values
#             y_predict_fulldata['y_pred'] = y_predict_fulldata['y_pred'].round()

#             # Plot the full data first
#             st.subheader('Full Data Plot')
#             fig_full = plot_full_data(data)
#             st.pyplot(fig_full)

#             # Plot the predicted values for full data predictions
#             st.subheader(f'Predicted Values for Full Data ({fh_full} Months)')
#             fig_predicted = plot_predicted_data(y_predict_fulldata, fh_full)
#             st.pyplot(fig_predicted)

#             # Display the predicted values with month and year only
#             y_predict_fulldata.index = y_predict_fulldata.index.strftime('%b %Y')
#             st.write('Predicted Values:')
#             st.write(y_predict_fulldata)

# except FileNotFoundError:
#     st.error("The specified file path is not valid. Please enter a valid file path.")
# except Exception as e:
#     st.error(f"An error occurred: {e}")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.time_series import TSForecastingExperiment
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Set page config
st.set_page_config(layout='wide', page_title='Economic Indicator (FRED Data)', page_icon=":chart_with_upwards_trend:")

# Access secrets from secrets.toml
file_path = st.secrets["data"]["file_path"]

# Function to load data
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    data.set_index('observation_date', inplace=True)
    return data

# Function to create and plot model predictions
def plot_predictions(train, fh, model_code='auto_arima'):
    exp = TSForecastingExperiment()
    exp.setup(data=train, fh=fh, fold=1, session_id=42)
    
    model = exp.create_model(model_code)
    final_model = exp.finalize_model(model)
    y_predict = exp.predict_model(final_model)
    
    return y_predict, final_model, exp

# Function to plot full data
def plot_full_data(data):
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Plot full data
    ax.plot(data.index, data['MRTSSM4451USS'], marker='o', linestyle='-', color='b', label='Actual')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Actual Values (Full Data)')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis to show month and year
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

    # Format y-axis to include commas for thousands separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    return fig

# Function to plot predicted data
def plot_predicted_data(y_predict, fh):
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Convert PeriodIndex to DatetimeIndex
    y_predict.index = y_predict.index.to_timestamp()
    
    # Plot predicted data
    y_pred_index = pd.date_range(start=y_predict.index[0], periods=fh, freq='MS')
    ax.plot(y_pred_index, y_predict['y_pred'], marker='o', linestyle='--', color='r', label='Predicted')

    # Set x-axis ticks to monthly intervals
    ax.set_xticks(y_pred_index)
    
    # Format x-axis to show month and year
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Predicted Values for {fh} Months')
    ax.legend()
    ax.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Format y-axis to include commas for thousands separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    return fig

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(time_index, actual_values, predicted_values, fh):
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot actual vs predicted values for the selected forecast horizon
    ax.plot(time_index, actual_values, marker='o', linestyle='-', color='b', label='Actual')
    ax.plot(time_index, predicted_values, marker='o', linestyle='--', color='r', label='Predicted')
    
    # Set x-axis ticks to monthly intervals
    ax.set_xticks(time_index)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Actual vs Predicted Values (Test Data for {fh} Months)')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis to show month and year
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Format y-axis to include commas for thousands separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Set y-axis ticks with a difference of 100
    ax.yaxis.set_major_locator(MultipleLocator(100))
    
    return fig

# Page title
st.title('Economic Indicator Forecast (FRED Data)')

try:
    data = load_data(file_path)

    # Display the first few rows of the dataframe and start/end dates
    st.write('Data preview:')
    st.write(data.head())
    st.write(f"Start Date: {data.index.min().strftime('%b %Y')}")
    st.write(f"End Date: {data.index.max().strftime('%b %Y')}")

    # Select the training and testing split
    st.sidebar.subheader('Select Tab')
    tab_selection = st.sidebar.radio('Select your tab:', ('Train Test split, validating data', 'Full data predictions'))

    if tab_selection == 'Train Test split, validating data':
        st.subheader('Train Test split, validating data')

        st.sidebar.subheader('Select Training and Testing Split')
        start_date = st.sidebar.date_input('Training start date', value=pd.to_datetime('1992-01-01'), min_value=pd.to_datetime('1992-01-01'))
        end_date = st.sidebar.date_input('Training end date', value=pd.to_datetime('2023-03-01'))
        
        # Convert start_date and end_date to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        train = data[(data.index >= start_date) & (data.index <= end_date)]
        test = data[data.index > end_date]

        max_fh = len(test)
        st.write(f'Maximum forecast horizon (fh) based on test data length: {max_fh} months')
        
        # Select the number of months to predict
        fh = st.slider('Number of months to predict:', min_value=1, max_value=max_fh, value=12)

        # Enter button to generate predictions
        if st.button('Enter'):
            # Plot the predictions for the selected forecast horizon
            st.subheader(f'Predictions for {fh} Months')
            y_predict, final_model, exp = plot_predictions(train, fh)

            actual_values = test['MRTSSM4451USS'][:fh].values
            predicted_values = y_predict['y_pred'].values

            # Round off the predicted values
            predicted_values = predicted_values.round()

            # Plot actual vs predicted values for the selected forecast horizon
            time_index = pd.date_range(start=end_date + pd.DateOffset(months=1), periods=fh, freq='MS')
            fig = plot_actual_vs_predicted(time_index, actual_values, predicted_values, fh)
            
            # Display the plot using st.pyplot()
            st.pyplot(fig)

            # Calculate and display MAPE for the selected forecast horizon
            mape = mean_absolute_percentage_error(actual_values, predicted_values)
            st.markdown(f'**Mean Absolute Percentage Error (MAPE) for {fh} months: {mape}**')

            # Display the actual and predicted values with month and year only
            result_df = pd.DataFrame({
                'Date': time_index.strftime('%b %Y'),
                'Actual Value': actual_values,
                'Predicted Value': predicted_values
            })
            st.write('Actual and Predicted Values:')
            st.write(result_df)

    elif tab_selection == 'Full data predictions':
        st.subheader('Full data predictions')

        # Select the number of months to predict for full data
        fh_full = st.slider('Number of months to predict for full data:', min_value=1, max_value=12, value=12)

        # Enter button to generate predictions
        if st.button('Enter for Full Data Predictions'):
            # Initialize TSForecastingExperiment for full data predictions
            exp = TSForecastingExperiment()
            exp.setup(data=data, fh=fh_full, fold=1, session_id=42)
            best_model = exp.create_model('auto_arima')
            final_model = exp.finalize_model(best_model)
            y_predict_fulldata = exp.predict_model(final_model)

            # Round off the predicted values
            y_predict_fulldata['y_pred'] = y_predict_fulldata['y_pred'].round()

            # Plot the full data first
            st.subheader('Full Data Plot')
            fig_full = plot_full_data(data)
            st.pyplot(fig_full)

            # Plot the predicted values for full data predictions
            st.subheader(f'Predicted Values for Full Data ({fh_full} Months)')
            fig_predicted = plot_predicted_data(y_predict_fulldata, fh_full)
            st.pyplot(fig_predicted)

            # Display the predicted values with month and year only
            y_predict_fulldata.index = y_predict_fulldata.index.strftime('%b %Y')
            st.write('Predicted Values:')
            st.write(y_predict_fulldata)

except FileNotFoundError:
    st.error("The specified file path is not valid. Please enter a valid file path.")
except Exception as e:
    st.error(f"An error occurred: {e}")

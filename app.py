import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

# Daten laden
data = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# Länderzuordnung
country_map = {
    'AT': 'Österreich', 'BE': 'Belgien', 'BG': 'Bulgarien', 'CH': 'Schweiz', 'CY': 'Zypern', 'CZ': 'Tschechien',
    'DE': 'Deutschland', 'DK': 'Dänemark', 'EE': 'Estland', 'ES': 'Spanien', 'FI': 'Finnland', 'FR': 'Frankreich',
    'GR': 'Griechenland', 'HR': 'Kroatien', 'HU': 'Ungarn', 'IE': 'Irland', 'IT': 'Italien', 'LT': 'Litauen',
    'LU': 'Luxemburg', 'LV': 'Lettland', 'MT': 'Malta', 'NL': 'Niederlande', 'NO': 'Norwegen', 'PL': 'Polen',
    'PT': 'Portugal', 'RO': 'Rumänien', 'SE': 'Schweden', 'SI': 'Slowenien', 'SK': 'Slowakei', 'GB': 'Vereinigtes Königreich'
}

# Relevante Spalten filtern
load_columns = [col for col in data.columns if col.endswith('_load_actual_entsoe_transparency')]
price_columns = [col for col in data.columns if col.endswith('_price_day_ahead')]

# Dropdown-Optionen erstellen
country_options_load = {country_map[col.split('_')[0]]: col for col in load_columns if col.split('_')[0] in country_map}
country_options_price = {country_map[col.split('_')[0]]: col for col in price_columns if col.split('_')[0] in country_map}

# Seitenleiste für Navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Wählen Sie einen Abschnitt", ("Daten filtern", "Vorhersage", "Komponenten", "Ländervergleich"))

# Funktion zur Visualisierung von Daten
def plot_interactive_data(data, title, ylabel, description):
    st.subheader(title)
    st.markdown(description)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['load'], mode='lines', name='Load'))
    fig.update_layout(
        title=title,
        xaxis_title='Datum',
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

if section == "Daten filtern":
    st.title("Daten filtern")
    st.write("In diesem Abschnitt können Sie die Daten nach einem bestimmten Zeitraum filtern und die täglichen Stromverbrauchsdaten in einem interaktiven Diagramm anzeigen.")
    
    # Land auswählen
    country = st.selectbox("Wählen Sie ein Land", list(country_options_load.keys()), index=list(country_options_load.keys()).index("Deutschland"))
    load_column = country_options_load[country]
    
    # Datumsbereich für Filterung
    start_date = st.date_input("Startdatum", pd.to_datetime("2015-01-01").date())
    end_date = st.date_input("Enddatum", pd.to_datetime("2022-01-01").date())
    
    # Zeitzoneninformationen entfernen
    data.index = data.index.tz_localize(None)
    filtered_data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))][[load_column]].dropna()
    filtered_data.columns = ['load']
    
    # Daten auf tägliche Auflösung setzen und gleitenden Durchschnitt berechnen
    filtered_data = filtered_data.resample('D').mean()
    window_size = 7  # Standardfenstergröße für den gleitenden Durchschnitt
    filtered_data['load'] = filtered_data['load'].rolling(window=window_size).mean()
    
    # Gefilterte Daten anzeigen
    st.subheader("Gefilterte Stromverbrauchsdaten")
    st.write(filtered_data)
    
    # Gefilterte Daten plotten
    plot_interactive_data(
        filtered_data,
        "Stromverbrauch in {}".format(country),
        "Verbrauch",
        "Dieses Diagramm zeigt den täglichen Stromverbrauch in {}. Der gleitende Durchschnitt wird zur besseren Sichtbarkeit der Trends verwendet.".format(country)
    )

elif section == "Vorhersage":
    st.title("Vorhersage des Stromverbrauchs und der Strompreise")
    st.write("In diesem Abschnitt sehen Sie die Vorhersage des Stromverbrauchs und der Strompreise basierend auf dem Prophet-Modell und die Berechnung des RMSE.")
    
    # Land auswählen
    country = st.selectbox("Wählen Sie ein Land", list(country_options_load.keys()), index=list(country_options_load.keys()).index("Deutschland"))
    load_column = country_options_load[country]
    price_column = country_options_price.get(country, None)
    
    # Daten vorbereiten
    country_data_load = data[[load_column]].resample('D').sum().dropna()
    country_data_load.columns = ['load']
    country_data_load = country_data_load[country_data_load['load'] > 0]
    country_data_load['log_load'] = np.log(country_data_load['load'])
    country_data_load = country_data_load.reset_index()
    country_data_load['utc_timestamp'] = country_data_load['utc_timestamp'].dt.tz_localize(None)
    country_data_load = country_data_load.rename(columns={'utc_timestamp': 'ds', 'log_load': 'y'})
    
    if price_column:
        country_data_price = data[[price_column]].resample('D').mean().dropna()
        country_data_price.columns = ['price']
        country_data_price = country_data_price[country_data_price['price'] > 0]
        country_data_price = country_data_price.reset_index()
        country_data_price['utc_timestamp'] = country_data_price['utc_timestamp'].dt.tz_localize(None)
        country_data_price = country_data_price.rename(columns={'utc_timestamp': 'ds', 'price': 'y'})
    
    # Train und Test splitten
    train_load = country_data_load[:int(0.8 * len(country_data_load))]
    test_load = country_data_load[int(0.8 * len(country_data_load)):]
    
    if price_column:
        train_price = country_data_price[:int(0.8 * len(country_data_price))]
        test_price = country_data_price[int(0.8 * len(country_data_price)):]
    
    # Prophet Modell für Verbrauch trainieren
    model_load = Prophet()
    model_load.fit(train_load)
    
    # Vorhersage für Verbrauch
    future_load = model_load.make_future_dataframe(periods=len(test_load))
    forecast_load = model_load.predict(future_load)
    
    # Gleitender Durchschnitt der Vorhersage
    forecast_load['yhat'] = forecast_load['yhat'].rolling(window=7).mean().fillna(method='bfill')
    
    # Vorhersage plotten
    st.subheader("Vorhersage des Stromverbrauchs")
    st.markdown("Dieses Diagramm zeigt die Vorhersage des Stromverbrauchs basierend auf historischen Daten. Der gleitende Durchschnitt wird zur besseren Sichtbarkeit der Trends verwendet.")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=train_load['ds'], y=train_load['y'], mode='lines', name='Train'))
    fig1.add_trace(go.Scatter(x=test_load['ds'], y=test_load['y'], mode='lines', name='Test'))
    fig1.add_trace(go.Scatter(x=forecast_load['ds'], y=forecast_load['yhat'], mode='lines', name='Forecast'))
    fig1.update_layout(
        title='Vorhersage des Stromverbrauchs',
        xaxis_title='Datum',
        yaxis_title='Verbrauch (log-transformiert)',
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # RMSE, MAE und R² für Verbrauch berechnen
    rmse_load = np.sqrt(mean_squared_error(test_load['y'], forecast_load['yhat'].iloc[-len(test_load):]))
    mae_load = mean_absolute_error(test_load['y'], forecast_load['yhat'].iloc[-len(test_load):])
    r2_load = r2_score(test_load['y'], forecast_load['yhat'].iloc[-len(test_load):])
    st.write(f'RMSE für Verbrauch: {rmse_load}')
    st.write(f'MAE für Verbrauch: {mae_load}')
    st.write(f'R² für Verbrauch: {r2_load}')
    
    if price_column:
        # Prophet Modell für Preise trainieren
        model_price = Prophet()
        model_price.fit(train_price)
        
        # Vorhersage für Preise
        future_price = model_price.make_future_dataframe(periods=len(test_price))
        forecast_price = model_price.predict(future_price)
        
        # Gleitender Durchschnitt der Vorhersage
        forecast_price['yhat'] = forecast_price['yhat'].rolling(window=7).mean().fillna(method='bfill')
        
                # Vorhersage für Preise plotten
        st.subheader("Vorhersage der Strompreise")
        st.markdown("Dieses Diagramm zeigt die Vorhersage der Strompreise basierend auf historischen Daten. Der gleitende Durchschnitt wird zur besseren Sichtbarkeit der Trends verwendet.")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=train_price['ds'], y=train_price['y'], mode='lines', name='Train'))
        fig2.add_trace(go.Scatter(x=test_price['ds'], y=test_price['y'], mode='lines', name='Test'))
        fig2.add_trace(go.Scatter(x=forecast_price['ds'], y=forecast_price['yhat'], mode='lines', name='Forecast'))
        fig2.update_layout(
            title='Vorhersage der Strompreise',
            xaxis_title='Datum',
            yaxis_title='Preis (€/MWh)',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # RMSE, MAE und R² für Preise berechnen
        rmse_price = np.sqrt(mean_squared_error(test_price['y'], forecast_price['yhat'].iloc[-len(test_price):]))
        mae_price = mean_absolute_error(test_price['y'], forecast_price['yhat'].iloc[-len(test_price):])
        r2_price = r2_score(test_price['y'], forecast_price['yhat'].iloc[-len(test_price):])
        st.write(f'RMSE für Strompreise: {rmse_price}')
        st.write(f'MAE für Strompreise: {mae_price}')
        st.write(f'R² für Strompreise: {r2_price}')

elif section == "Ländervergleich":
    st.title("Ländervergleich")
    st.write("In diesem Abschnitt können Sie den Stromverbrauch mehrerer Länder vergleichen.")
    
    # Länder auswählen
    countries = st.multiselect("Wählen Sie Länder", list(country_options_load.keys()), default=["Deutschland"])
    
    # Datumsbereich für Filterung
    start_date = st.date_input("Startdatum", pd.to_datetime("2015-01-01").date())
    end_date = st.date_input("Enddatum", pd.to_datetime("2022-01-01").date())
    
    fig = go.Figure()
    
    for country in countries:
        load_column = country_options_load[country]
        
        # Zeitzoneninformationen entfernen
        data.index = data.index.tz_localize(None)
        filtered_data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))][[load_column]].dropna()
        filtered_data.columns = ['load']
        
        # Daten auf tägliche Auflösung setzen und gleitenden Durchschnitt berechnen
        filtered_data = filtered_data.resample('D').mean()
        window_size = 7  # Standardfenstergröße für den gleitenden Durchschnitt
        filtered_data['load'] = filtered_data['load'].rolling(window=window_size).mean()
        
        # Daten zum Plot hinzufügen
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['load'], mode='lines', name=country))

    fig.update_layout(
        title='Stromverbrauch im Ländervergleich',
        xaxis_title='Datum',
        yaxis_title='Verbrauch',
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

elif section == "Komponenten":
    st.title("Komponenten der Zeitreihe")
    st.write("In diesem Abschnitt sehen Sie die verschiedenen Komponenten der Zeitreihe wie den Trend, wöchentliche und jährliche Muster.")
    
    # Land auswählen
    country = st.selectbox("Wählen Sie ein Land", list(country_options_load.keys()), index=list(country_options_load.keys()).index("Deutschland"))
    load_column = country_options_load[country]
    
    # Daten vorbereiten
    country_data = data[[load_column]].resample('D').sum().dropna()
    country_data.columns = ['load']
    country_data = country_data[country_data['load'] > 0]
    country_data['log_load'] = np.log(country_data['load'])
    country_data = country_data.reset_index()
    country_data['utc_timestamp'] = country_data['utc_timestamp'].dt.tz_localize(None)
    country_data = country_data.rename(columns={'utc_timestamp': 'ds', 'log_load': 'y'})
    
    # Prophet Modell trainieren
    model = Prophet()
    model.fit(country_data)
    
    # Vorhersage
    forecast = model.predict(country_data)
    
    # Komponenten plotten
    st.subheader("Komponenten der Zeitreihe")
    st.markdown("Dieses Diagramm zeigt die verschiedenen Komponenten der Zeitreihe wie den Trend, wöchentliche und jährliche Muster.")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)


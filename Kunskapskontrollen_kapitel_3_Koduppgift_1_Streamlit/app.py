#Kapitel 3 kodupgift 15

#Den här applikationen använder machine learning för att prediktera priset på en bil. Den beroende variabeln är Price. 
# Appen läser in data från en CSV-fil, tränar en modell, utvärderar modellen med RMSE, låter användaren skriva in biluppgifter och visar ett predikterat pris.

#c) Modellen skulle kunna användas i verkligheten för att uppskatta vad en bil är värd. 
# Till exempel skulle den kunna användas på hemsidor där man säljer bilar eller av bilhandlare för att få en snabb prisbedömning. 
# Den kan vara till hjälp både för säljare som vill sätta ett rimligt pris och för köpare som vill veta om priset är rimligt. 
# Samtidigt är modellen ganska enkel och tar inte hänsyn till alla faktorer som kan påverka priset, som till exempel bilens skick eller extra utrustning. 
# Därför skulle man i verkligheten behöva mer data och mer avancerade modeller för att få mer exakta resultat.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

#Läs in data 
df = pd.read_csv("car_price_dataset.csv", sep=";")

#Dela up datan i x och y 
X = df.drop("Price", axis=1)
y = df["Price"]

# Gör om textkolumner till dummyvariabler ( text till siffror 1 och 0), (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Dela upp data i träningsdata och testdata 80% träning, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Träna linjär regressionsmodell
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediktion på testdata
y_pred = lr.predict(X_test)

# Beräkna RMSE, ju lägre ju bättre 
rmse = root_mean_squared_error(y_test, y_pred)

# Diagram
plt.style.use("seaborn-v0_8")

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(y_test, y_pred, alpha=0.6)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=2)

ax.set_xlabel("Faktiskt pris")
ax.set_ylabel("Predikterat pris")
ax.set_title("Prediktion vs verklighet")

plt.tight_layout()

# Streamlit-app
st.sidebar.title("Meny")
choice = st.sidebar.radio("Gå till", ["Hem", "Data och modell", "Prediktera pris"])

# Hem
if choice == "Hem":
    st.title("Bilpris-prediktion")
    
    st.write("""
    Den här applikationen använder machine learning för att prediktera priset på en bil.
    Den beroende variabeln är **Price**.
    """)
    
    st.write("""
    Appen gör följande:
    - läser in data från en CSV-fil
    - tränar en modell
    - utvärderar modellen med RMSE
    - låter användaren skriva in biluppgifter
    - visar ett predikterat pris
    """)

# Data och modell
elif choice == "Data och modell":
    st.title("Data och modell")

    if st.checkbox("Visa dataset"):
        st.dataframe(df)

    st.subheader("Modellens resultat")
    st.write(f"RMSE: {rmse:.2f}")

    st.subheader("Diagram")
    st.pyplot(fig)


# Prediktera pris
elif choice == "Prediktera pris":
    st.title("Prediktera bilpris")

    brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
    model_name = st.selectbox("Model", sorted(df["Model"].unique()))
    year = st.number_input("Year", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()), value=int(df["Year"].median()))
    engine_size = st.number_input("Engine Size", min_value=float(df["Engine_Size"].min()), max_value=float(df["Engine_Size"].max()), value=float(df["Engine_Size"].median()))
    fuel_type = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].unique()))
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
    mileage = st.number_input("Mileage", min_value=int(df["Mileage"].min()), max_value=int(df["Mileage"].max()), value=int(df["Mileage"].median()))
    doors = st.number_input("Doors", min_value=int(df["Doors"].min()), max_value=int(df["Doors"].max()), value=int(df["Doors"].median()))
    owner_count = st.number_input("Owner Count", min_value=int(df["Owner_Count"].min()), max_value=int(df["Owner_Count"].max()), value=int(df["Owner_Count"].median()))

    # Skapa en DataFrame av användarens inmatning
    user_input = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Year": [year],
        "Engine_Size": [engine_size],
        "Fuel_Type": [fuel_type],
        "Transmission": [transmission],
        "Mileage": [mileage],
        "Doors": [doors],
        "Owner_Count": [owner_count]
    })

    # hanterar kategoriska variabler på samma sätt som vid träning (dummy-variabler)
    user_input = pd.get_dummies(user_input)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    if st.button("Visa prediktion"):
        prediction = lr.predict(user_input)
        st.success(f"Predikterat pris: {prediction[0]:.0f}")
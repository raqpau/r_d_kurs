import streamlit as st
import pandas as pd
import joblib

# === Funkcja predykcji ===
def predict_price(area_m2, rooms, photos):
    # Wczytanie modelu tylko raz (np. Random Forest z Adresowo)
    model = joblib.load("model_random_forest_adresowo_lodz.pkl")
    X_new = pd.DataFrame([[area_m2, rooms, photos]],
                         columns=["area_m2", "rooms", "photos"])
    return model.predict(X_new)[0]

# === Główna logika aplikacji ===
def main():
    st.title("Predykcja ceny mieszkania (Adresowo)")
    st.write("Podaj dane mieszkania, aby uzyskać szacowaną cenę:")

    # Komponenty UI
    area = st.number_input("Powierzchnia (m²)", min_value=10.0, max_value=300.0, value=50.0)
    rooms = st.slider("Liczba pokoi", 1, 6, 3)
    photos = st.number_input("Liczba zdjęć", 0, 50, 10)

    if st.button("Oblicz cenę"):
        price = predict_price(area, rooms, photos)
        st.success(f"Szacowana cena: {price:,.0f} zł")

# === Punkt wejścia ===
if __name__ == "__main__":
    main()
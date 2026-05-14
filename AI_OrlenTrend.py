import os
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Wczytuje dane z pliku .env do środowiska Pythona
load_dotenv()

# Zamiast OpenAI, używamy Groq (który jest kompatybilny z biblioteką OpenAI)
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

ticker = "^GSPC"

# Pobierz dane z ostatniego tygodnia
dane = yf.download(ticker, period="1y", interval="1d")

# 3. Obliczanie średnich kroczących (np. 5-dniowa i 20-dniowa)
# .rolling(window=X) oblicza średnią z X ostatnich wierszy (dni)
dane['SMA_5'] = dane['Close'].rolling(window=5).mean()
dane['SMA_20'] = dane['Close'].rolling(window=20).mean()

# 4. Filtrowanie danych do ostatniego tygodnia (5 dni roboczych), aby wykres był czytelny
dane_tydzien = dane.tail(30)

# 5. Tworzenie wykresu
plt.figure(figsize=(10, 6))

# Rysowanie linii ceny zamknięcia oraz średnich kroczących
plt.plot(dane_tydzien.index, dane_tydzien['Close'], label='Cena S&P 500', color='blue', marker='o', linewidth=2)
plt.plot(dane_tydzien.index, dane_tydzien['SMA_5'], label='Średnia 5-dniowa (SMA 5)', color='orange', linestyle='--')
plt.plot(dane_tydzien.index, dane_tydzien['SMA_20'], label='Średnia 20-dniowa (SMA 20)', color='green', linestyle='-.')

# Dodawanie tytułów, etykiet i legendy
plt.title('Trend S&P 500 z ostatniego miesiąca wraz ze średnimi kroczącymi', fontsize=14, fontweight='bold')
plt.xlabel('Data', fontsize=12)
plt.ylabel('Wartość indeksu (USD)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')

# Automatyczne dopasowanie i wyświetlenie wykresu
plt.tight_layout()
plt.show()

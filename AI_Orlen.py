import os
import yfinance as ticker_data
from crewai import Agent, Task, Crew, Process

# Zamiast OpenAI, używamy Groq (który jest kompatybilny z biblioteką OpenAI)
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.3-70b-versatile"
os.environ["OPENAI_API_KEY"] = ""

def get_orlen_price(ticker="PKN.WA"):
    """Funkcja pomocnicza do pobierania aktualnych danych z GPW"""
    stock = ticker_data.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="2d")
    
    current_price = info.get('regularMarketPrice') or hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[0]
    change = ((current_price - prev_close) / prev_close) * 100
    
    return {
        "price": round(current_price, 2),
        "change_pct": round(change, 2),
        "volume": info.get('regularMarketVolume', 'N/A')
    }

# Pobieramy dane przed uruchomieniem agenta, aby miał świeży kontekst
orlen_stats = get_orlen_price()

# 1. Definicja Agenta
analyst = Agent(
    role='Starszy Analityk GPW',
    goal=f'Monitorowanie spółki Orlen i raportowanie istotnych zmian kursu.',
    backstory="""Specjalizujesz się w polskim sektorze energetycznym. 
    Potrafisz odróżnić zwykły szum rynkowy od ważnych sygnałów technicznych.""",
    allow_delegation=False,
    verbose=True
)

# 2. Definicja Zadania
monitoring_task = Task(
    description=f"""Przeanalizuj bieżące dane dla Orlen (PKN.WA):
    - Aktualna cena: {orlen_stats['price']} PLN
    - Zmiana dzienna: {orlen_stats['change_pct']}%
    - Wolumen: {orlen_stats['volume']}
    
    Jeśli zmiana przekracza 1.5%, zidentyfikuj to jako zdarzenie 'Wysokiej Priorytetowości'. 
    Przygotuj raport dla inwestora, biorąc pod uwagę kontekst cenowy.""",
    agent=analyst,
    expected_output="Krótki, konkretny raport (max 3 punkty) z rekomendacją działania."
)

# 3. Formowanie Zespołu (Crew)
orlen_crew = Crew(
    agents=[analyst],
    tasks=[monitoring_task],
    process=Process.sequential
)

# 4. Start
print(f"--- Uruchamianie monitoringu dla Orlen ---")
result = orlen_crew.kickoff()

print("\n\n########################")
print("## RAPORT AGENTA AI ##")
print("########################\n")
print(result)

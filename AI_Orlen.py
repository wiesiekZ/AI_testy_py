import os
import yfinance as ticker_data
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool  # Importujemy dekorator tool
from dotenv import load_dotenv

# Wczytuje dane z pliku .env do środowiska Pythona
load_dotenv()

# Zamiast OpenAI, używamy Groq (który jest kompatybilny z biblioteką OpenAI)
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from datetime import datetime

# Pobieramy aktualną datę systemową
today = datetime.now().strftime("%Y-%m-%d")

def get_real_orlen_data():
    ticker = "PKN.WA"
    stock = yf.Ticker(ticker)
    # Pobieramy dane z ostatniego dnia sesyjnego
    df = stock.history(period="1d")
    
    if df.empty:
        return "Błąd: Nie udało się pobrać danych z giełdy."
    
    latest_price = df['Close'].iloc[-1]
    prev_close = stock.info.get('previousClose', 'N/A')
    
    return f"""
    DANE ZWERYFIKOWANE (Źródło: Yahoo Finance):
    Spółka: ORLEN S.A.
    Aktualna cena (zamknięcie): {latest_price:.2f} PLN
    Poprzednie zamknięcie: {prev_close} PLN
    Data odczytu: {today}
    """

# Wywołujemy funkcję przed agentem
actual_market_data = get_real_orlen_data()

# 1. Definiujemy narzędzie w sposób, który Pydantic zaakceptuje
@tool("search_tool")
def search_tool(question: str):
    """Przeszukuje internet w poszukiwaniu informacji na podany temat."""
    return DuckDuckGoSearchRun().run(question)

# 2. Definicja Agentów (zmieniamy sposób przypisania narzędzia)
data_analyst = Agent(
    role='Analityk Techniczny GPW',
    goal='Raportowanie wyłącznie na podstawie dostarczonych danych liczbowych.',
    backstory="""Jesteś surowym analitykiem. 
    ZASADA 1: Jeśli nie masz danych liczbowych, napisz 'BRAK DANYCH'. 
    ZASADA 2: Nigdy, pod żadnym pozorem nie zgaduj ani nie wymyślaj cen. 
    ZASADA 3: Operuj tylko na danych oznaczonych jako 'DANE ZWERYFIKOWANE'.""",
    verbose=True
)

news_researcher = Agent(
    role='Dziennikarz Śledczy Biznesu',
    goal='Znalezienie najnowszych wiadomości o spółce Orlen.',
    backstory='Jesteś najlepszym researcherem w kraju.',
    tools=[search_tool], # Teraz używamy naszej funkcji z dekoratorem
    verbose=True,
    allow_delegation=False
)


# 3. Definicja Zadań
task1 = Task(
    description=f"""Używając poniższych danych:
    {actual_market_data}
    
    Przedstaw krótki raport. Jeśli dane wskazują na rok inny niż 2026, 
    poinformuj o błędzie źródła i nie kontynuuj analizy.""",
    expected_output="Raport oparty wyłącznie na faktach liczbowych.",
    agent=data_analyst
)

task2 = Task(
    description='Wyszukaj w internecie wiadomości z ostatnich 24h dotyczące spółki Orlen (PKN.WA) lub polskiego sektora paliwowego.',
    expected_output='Podsumowanie 3 najważniejszych newsów wraz z oceną sentymentu (pozytywny/negatywny/neutralny).',
    agent=news_researcher
)

task3 = Task(
    description='Na podstawie danych liczbowych i newsów, napisz finałowy raport: "Dlaczego kurs się zmienia i co może wydarzyć się jutro?".',
    expected_output='Kompleksowy raport końcowy dla inwestora.',
    agent=data_analyst # Główny analityk składa wszystko w całość
)

# 4. Uruchomienie Procesu (Sekwencyjnego - jeden agent czeka na drugiego)
orlen_crew = Crew(
    agents=[data_analyst, news_researcher],
    tasks=[task1, task2, task3],
    process=Process.sequential
)

print("--- Uruchamiam zaawansowaną analizę Orlenu ---")
result = orlen_crew.kickoff()
print("\n\n########################")
print(result)

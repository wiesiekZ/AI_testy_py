import os
import yfinance as ticker_data
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool  # Importujemy dekorator tool

# Zamiast OpenAI, używamy Groq (który jest kompatybilny z biblioteką OpenAI)
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.3-70b-versatile"
os.environ["OPENAI_API_KEY"] = ""


# 1. Definiujemy narzędzie w sposób, który Pydantic zaakceptuje
@tool("search_tool")
def search_tool(question: str):
    """Przeszukuje internet w poszukiwaniu informacji na podany temat."""
    return DuckDuckGoSearchRun().run(question)

# 2. Definicja Agentów (zmieniamy sposób przypisania narzędzia)
data_analyst = Agent(
    role='Analityk Techniczny GPW',
    goal='Analiza kursu Orlenu na podstawie danych liczbowych.',
    backstory='Jesteś ekspertem od liczb i wykresów.',
    verbose=True,
    allow_delegation=False # Wyłączamy delegację, by uniknąć zapętleń
)

news_researcher = Agent(
    role='Dziennikarz Śledczy Biznesu',
    goal='Znalezienie najnowszych wiadomości o spółce Orlen.',
    backstory='Jesteś najlepszym researcherem w kraju.',
    tools=[search_tool], # Teraz używamy naszej funkcji z dekoratorem
    verbose=True,
    allow_delegation=False
)


from datetime import datetime

# Pobieramy aktualną datę systemową
today = datetime.now().strftime("%Y-%m-%d")

# 3. Definicja Zadań
task1 = Task(
    description=f"Dziś jest {today}. Pobierz AKTUALNE notowania dla Orlen (PKN.WA) z dzisiejszej sesji. Zignoruj dane historyczne z lat 2024-2025. Interesuje mnie tylko cena z ostatnich 24 godzin.",
    expected_output="Raport o cenie Orlenu z dzisiejszego dnia.",
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

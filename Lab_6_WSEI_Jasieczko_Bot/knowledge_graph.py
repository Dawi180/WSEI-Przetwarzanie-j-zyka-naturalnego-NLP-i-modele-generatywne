import os
from datetime import datetime
import wikipedia
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
wikipedia.set_user_agent("WSEIBot_NLP_Lab4/1.0 (student@wsei.edu.pl)")


# ==========================================
# 1. NAMED ENTITY LINKING (NEL) - WIKIPEDIA
# ==========================================
def link_entity_wikipedia(entity_name, lang="pl"): # Domyślnie na polski
    wikipedia.set_lang(lang)
    entity_name = entity_name.strip() # Usunięcie złośliwych, niewidzialnych spacji!
    
    try:
        search_results = wikipedia.search(entity_name, results=3)
        if not search_results:
            # Fallback na angielski
            wikipedia.set_lang("en")
            search_results = wikipedia.search(entity_name, results=3)
            if not search_results:
                return None

        candidates = []
        for res in search_results:
            try:
                # Usunięto problematyczny auto_suggest=False
                page = wikipedia.page(res)
                candidates.append({
                    "title": page.title,
                    "url": page.url,
                    "summary": page.summary[:150] + "..."
                })
            except Exception:
                # Jak Wikipedia nie jest pewna co zwrócić, pomijamy hasło
                continue
                
        return candidates if candidates else None
    except Exception as e:
        print(f"Błąd wyszukiwania Wikipedii dla {entity_name}: {e}")
        return None

# ==========================================
# 2. GENERATOR GRAFÓW WIEDZY
# ==========================================
def generate_knowledge_graph(entities, text="Dokument"):
    """
    Generuje graf wiedzy z wyciągniętych entitetów za pomocą networkx.
    Zapisuje graf jako plik PNG.
    """
    # Upewniamy się, że folder z Lab 4 istnieje
    plots_dir = "lab4plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    G = nx.Graph()
    
    # Dodajemy węzeł centralny symbolizujący cały tekst
    central_node = "TEXT"
    G.add_node(central_node)
    
    # W prawdziwym systemie użylibyśmy analizy zależności (Dependency Parsing), 
    # aby znaleźć dokładne relacje (np. "założył", "pracuje w").
    # Dla naszego bota zbudujemy graf pokazujący połączenie entitetów z tekstem i ich TYPY.
    
    for ent_text, ent_type in entities:
        G.add_node(ent_text)
        # Krawędź między tekstem a entitetem, z etykietą typu (np. PERSON, ORG)
        G.add_edge(central_node, ent_text, label=ent_type)
    
    # Ustawienia rysowania grafu
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42) # Rozkład przestrzenny węzłów
    
    # Rysowanie węzłów i linii
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', edge_color='gray')
    
    # Rysowanie etykiet na liniach (czyli naszych typów entitetów)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    # Zapis do pliku ze znacznikiem czasu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(plots_dir, f"knowledge_graph_{timestamp}.png")
    
    plt.title("Graf Wiedzy (Knowledge Graph)")
    plt.savefig(filepath)
    plt.close()
    
    return filepath
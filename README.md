# Système Multi-Agent pour Support Client E-Commerce

Ce projet consiste à développer un système multi-agent intelligent dédié au support client d'une plateforme E-Commerce. Le système utilise l'orchestration de **LangGraph** et une architecture **Agentic RAG** (Retrieval-Augmented Generation) pour répondre aux requêtes des utilisateurs de manière autonome et précise. Il intègre également un mécanisme **Human-in-the-Loop** pour sécuriser les opérations sensibles.

##  Fonctionnalités Principales
* **Classification des Intentions :** Extraction automatique de l'intention du client et de l'identifiant de la commande (`order_id`) via un prompt Few-Shot structuré en JSON.
* **RAG Agentique :** Validation décisionnelle basée sur l'interrogation d'un index vectoriel local (`ChromaDB`) contenant les règles de retour et de remboursement de l'entreprise (`politique_remboursement.txt`).
* **Contrôle Humain (Human-in-the-Loop) :** Interruption automatique du graphe d'états pour validation par un opérateur humain dès qu'une action de remboursement (`refund`) est détectée.
* **Interface Graphique :** Une application web interactive développée avec **Streamlit** permettant de suivre en temps réel le raisonnement des agents et de gérer les flux d'approbation.

##  Architecture du Graphe (LangGraph)
Le système s'appuie sur un `StateGraph` persistant qui gère les transitions entre les nœuds suivants :
1. **Classifier Node :** Identifie l'intention et extrait l'ID.
2. **RAG Verifier Node :** Extrait les segments de texte pertinents et analyse la conformité avec le LLM Llama 3.3 70B via Groq.
3. **Validation Humaine :** Point d'interruption conditionnel si `action_required == "refund"`.
4. **Resolver Node :** Génère la solution technique adéquate.
5. **Responder Node :** Formule la réponse finale destinée au client.

##  Installation et Configuration

### 1. Cloner le projet
```bash
git clone [https://github.com/fatimazahramassane/E-Commerce_FWS.git](https://github.com/fatimazahramassane/E-Commerce_FWS.git)
cd E-Commerce_FWS
pip install -r requirements.txt```

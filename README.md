
# 📊 Portfolio Analytics – App locale (Streamlit)

Questa è una versione **app** del tuo codice Python/Jupyter, pronta per girare in locale.

## ▶️ Avvio rapido (Windows / macOS / Linux)

1. **Crea un virtual environment** (consigliato):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Avvia l'app**:
   ```bash
   streamlit run app_streamlit.py
   ```
   Si aprirà il browser su `http://localhost:8501` (se non si apre, copia/incolla l'URL nella barra indirizzi).

## 📁 Struttura

```
portfolio_core.py     # Tutta la logica di calcolo/metriche/ottimizzazione
app_streamlit.py      # Interfaccia utente Streamlit
requirements.txt
README.md
```

## 🧱 Note tecniche

- La UI originale basata su `ipywidgets` non è adatta a diventare un eseguibile desktop.
  Qui abbiamo portato la UI su **Streamlit**, che è perfetta per app locali semplici:
  gira in locale, non richiede server esterni e si controlla dal browser.
- I grafici sono in `matplotlib` e vengono mostrati con `st.pyplot`.
- Il download dati usa `yfinance`: serve una connessione Internet per scaricare le serie storiche.
- Metriche incluse: Sharpe, Sortino, Calmar/MAR, Burke, Sterling, Kappa(3), Rachev, ES/CVaR 95/99, Omega,
  drawdown, contributi al rischio (vol & CVaR).

## 🧪 Consigli operativi

- Se noti errori di “Ticker non trovato”, verifica il simbolo su Yahoo Finance.
- Se vuoi fissare i pesi, inserisci la lista (che sommi a 1.0) nel box **Pesi**.
- L’ottimizzazione supporta: `max_sharpe`, `max_sortino`, `min_cvar95`, `min_cvar99`.
  Puoi applicare **cap settoriali** tramite due campi di testo (mappa e cap).

## 📦 (Opzionale) Crea un lanciatore/collegamento

- **Windows**: crea un file `run.bat` con:
  ```bat
  @echo off
  call .venv\Scripts\activate
  streamlit run app_streamlit.py
  ```
  Poi fai doppio click su `run.bat`.

- **macOS/Linux**: crea `run.sh` (poi `chmod +x run.sh`):
  ```bash
  #!/usr/bin/env bash
  source .venv/bin/activate
  streamlit run app_streamlit.py
  ```

## 🧊 (Avanzato) Creare un eseguibile

Creare un vero **.exe / .app** per un'app Streamlit non è ufficialmente supportato e può richiedere
workaround (ad es. PyInstaller + wrapper che avvia Streamlit). Il metodo più semplice per uso personale
resta il lanciatore sopra. Se vuoi comunque il bundle, possiamo preparare un wrapper `run_app.py`
che avvia Streamlit programmaticamente e provare a "freezarlo" con PyInstaller.

Se vuoi una **GUI desktop nativa** (PyQt/PySide/Tkinter), si può fare, ma richiede più lavoro
per trasporre la UI what‑if e i grafici. Dimmi se preferisci questa strada e preparo un prototipo.

---

Buon lavoro! 🚀

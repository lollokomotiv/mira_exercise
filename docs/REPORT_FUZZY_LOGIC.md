# REPORT_FUZZY_LOGIC

- **Obiettivo**: `src/data_engineering/build_xref_all.py` costruisce tabelle di cross-reference tra giocatori e club provenienti da StatsBomb e Transfermarkt, usando fuzzy matching controllato per ridurre gli errori di associazione.

- **Preparazione dati**:
  - Normalizza i nomi (`normalize_name`) tramite lower-case, rimozione degli accenti e della punteggiatura per confronti coerenti anche quando i dataset hanno formattazioni diverse.
  - Crea varianti sui soprannomi (`NICK_MAP`) così da coprire alias noti (es. Dani → Daniel) e combina nick aggiuntivi lato funzione principale per aumentare il recall senza aprire troppo il ventaglio di candidati.
  - Enfatizza i cognomi (`_surname_tokens`) che sono il segnale più stabile tra campioni internazionali; da qui la scelta di aggiungere bonus specifici in fase di scoring.

- **Matching pipeline**:
  - Carica i CSV di partenza e ne filtra/arricchisce le colonne chiave (nome, nazione, club, posizione).
  - Organizza i giocatori Transfermarkt per nazione, restringendo il set di candidati e migliorando le performance del fuzzy matching. Se la nazione non è disponibile, torna all’intero dataset per evitare falsi negativi.
  - Valuta i candidati calcolando uno score ibrido (`_score`) che usa RapidFuzz quando presente (per una misura più robusta su stringhe rumorose) o `difflib` come fallback durevole.
  - Applica bonus/pene contestuali:
    - +0.08 se il cognome coincide, perché l’omonimia sul cognome è particolarmente affidabile.
    - +0.06 se la nazione combacia; scelta conservativa per non distorcere i punteggi quando i giocatori cambiano federazione.
    - Bonus proporzionale sull’overlap dei token (+0.05 * ratio) per favorire combinazioni con più parti del nome in comune.
    - +0.03 quando il club Transfermarkt contiene il nome della squadra StatsBomb, permettendo di distinguere meglio giocatori con nomi simili nello stesso campionato.
    - −0.03 in caso di nomi molto corti (<5 caratteri) per contrastare i falsi positivi dovuti a stringhe insignificanti.
  - Limita lo score finale a [0,1] e traccia diagnostiche (`score_raw`, `margin`, `second_best_name`) per audit manuali.

- **Classificazione e soglie**:
  - Determina il metodo (`exact`, `high`, `fuzzy`, `low`) in base allo score e al margine rispetto al secondo candidato migliore, così da dare una misura di confidenza e ridurre match ambigui.
  - `threshold_exact` e `threshold_high` sono fissati rispettivamente a 0.95 e 0.90 per favorire precisione sui casi accettati automaticamente, mentre `threshold_fuzzy` a 0.81 recupera accoppiamenti plausibili da sottoporre a revisione.
  - Produce tre viste: `player_xref_all_full.csv` (diagnostica completa), `player_xref_all_strict.csv` (solo match ≥0.95) e `player_xref_all_review_needed.csv` (tutto il resto) per bilanciare automazione e controllo manuale.

- **Club cross-reference**:
  - `build_club_xref` normalizza i nomi club Transfermarkt, generando un alias pronto per incroci successivi.

- **Robustezza & logging**:
  - L’uso di `Path` tutela la portabilità del percorso file.
  - I check di esistenza dei CSV e i messaggi `⚠️/✅` aiutano a diagnosticare rapidamente input mancanti o run parziali.

- **Considerazioni future**:
  - Ampliare i nickname e utilizzare feature aggiuntive (es. data di nascita) può aumentare la precisione nelle federazioni con molti omonimi.
  - Valutare l’introduzione di punteggi adattivi basati sulla distribuzione dei match per nazione per gestire dataset con densità molto diversa di giocatori.

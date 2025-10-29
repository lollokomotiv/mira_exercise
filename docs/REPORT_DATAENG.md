# Data Engineering Guide — EURO24 & National Tournaments (since 2018)

Questo documento spiega in modo discorsivo come ho costruiro i database SQLite usati dal modeling, quali scelte ho preso sul data model e perché. L’obiettivo è un DB piccolo, coerente e “query‑friendly” che faccia da cerniera tra i dati grezzi e i modelli.

In sintesi: prendiamo eventi/match da StatsBomb (filtrando per EURO24), li normalizziamo in CSV, definiamo le finestre di transfer “post‑torneo”, filtriamo le transazioni Transfermarkt dentro quelle finestre e colleghiamo i giocatori tra i due mondi tramite cross‑reference. Il risultato è un DB unico per EURO24 e un DB storico “all tournaments” che serve in fase di creazione del modello predittivo.

## Perché questo Data Model

- **Focus sul comportamento in torneo**: per stimare la fee “post torneo” vogliamo usare prestazioni in contesto nazionale (non stagioni intere di club). StatsBomb è la fonte più consistente e standardizzata per eventi e statistiche di torneo.
- **Finestra temporale realistica**: i trasferimenti dopo i tornei avvengono in un periodo relativamente concentrato. Definire una finestra ci permette di evitare leakage (es. fee influenzate da eventi successivi) e di rendere paragonabili i casi.
- **SQLite monolitico**: un file unico, versionabile, ripetibile e facile da interrogare. È sufficiente per il perimetro del progetto e minimizza la complessità di infrastruttura.
- **Cross‑reference esplicito (per collegare i PLAYERS da StatsBomb a Transfermarkt)**: si effettua una *fuzzy logic* per collegare i giocatori delle due piattaforme. Si tiene due mapping (strict e all) perché in un contesto reale conviene avere sia un match ad alta confidenza sia una copertura ampia per EDA o join “best effort”.

## Fonti e cosa manteniamo

- **StatsBomb (CSV “processed”)**: `matches`, `teams`, `players`, `player_match_stats`, `player_appearances`, `events_meta`, `referees`, `venues`.
- **Transfermarkt**: `transfers.csv` filtrato nelle finestre, più `tm_players.csv` per anagrafica (data di nascita, piede, altezza) e market value post‑transfer.
- **Cross‑reference**: `player_xref` (strict) e `player_xref_all` (wider), oltre a `club_xref` (minimale, solo alias).
- **Valuations storiche**: teniamo il CSV dei market value nel raw (per tagli diversi), ma non lo materializziamo dentro il DB per non vincolare l’analisi ad un solo cutoff.

## Come costruiamo i DB (passo‑passo)

1) **Estrazione StatsBomb (nazionali), normalizzazione JSON→CSV**
   - Multi‑torneo (since 2018): `src/pipeline/run_all_tournaments.py`
   - EURO24: `src/pipeline/run_euro24.py`
   - Perché Polars: è rapido e memory‑friendly per l’ingest, mentre la modellazione resta in pandas.

2) **Definizione delle finestre “post‑torneo”**
   - EURO24: finestra fissa 2024‑07‑15 ≤ date < 2024‑10‑01 (mercato estivo post EURO).
   - Storico multi‑torneo: calcoliamo la data della finale per ciascun torneo e creiamo una finestra di 90 giorni dopo (vedi `src/data_engineering/windows.py`).
   - Razionale: 90 giorni coprono la quasi totalità delle trattative estive legate alla visibilità del torneo, mantenendo coerenza tra competizioni diverse.

3) **Filtriamo i trasferimenti Transfermarkt nelle finestre**
   - Generiamo `transfers_all.csv` (storico) o `transfers_euro24.csv` (EURO24), mantenendo i campi essenziali: fee, MV post (`market_value_in_eur`), da/verso club, data, nome giocatore.

4) **Cross‑reference giocatori con FUZZY LOGIC (StatsBomb ↔ Transfermarkt)**
   - `player_xref` (strict): match ad alta confidenza, preferito nei join “di qualità”.
   - `player_xref_all` (allargato): usato per aumentare copertura, EDA e viste “_all”.
   - Perché due varianti: separare qualità da copertura aiuta a non inquinare i dataset supervisati ma garantisce esplorazione completa.

5) **Materializzazione schema in SQLite**
   - Script: `src/data_engineering/build_schema_euro24.py` e `build_schema_all.py`.
   - Tabelle principali: teams, players, matches, player_match_stats, player_appearances, events_meta, transfers, tm_players, xref (strict/all), club_xref.
   - Indici: su chiavi e campi di filtro comuni (date, match_id, player_id, TM ids) per query rapide.
   - Viste “di lavoro”: per EURO24 e per NT (storico) creiamo viste che uniscono giocatori SB con transfers in finestra, in varianti strict e all.

6) **Per‑90 e feature derivate**
   - Nel DB manteniamo i totali grezzi per semplicità; la conversione per‑90 avviene in fase di feature building (modello), con accortezze come:
     - cap minimo 180’ per stabilità dei ratei;
     - winsorization delle code (1–2%) per ridurre l’effetto outlier;
     - gestione robusta dei missing (conversione numerica, imputazione). 

## Schema (a colpo d’occhio)

- `teams(team_id PK, team_name)`
- `players((player_id, team_id) PK, player_name, team_name, primary_position, jersey_number)`
- `matches(match_id PK, match_date, …, home/away team id/name, score, stage, referee, venue)`
- `player_match_stats(match_id, team_id, player_id, minutes, goals, shots, xg, xa, tackles, dribbles, pressures, interceptions, …)`
- `player_appearances(appearance_id PK, match_id, team_id, player_id, started, minutes, position_played, jersey_number)`
- `events_meta(match_id, idx, id, period, minute, second, team_id, player_id, type, …)`
- `tm_players(player_id, date_of_birth, position, foot, height_in_cm, …)`
- `transfers(player_id TM, transfer_date, from/to club, transfer_fee, market_value_in_eur, player_name)`
- `player_xref(statsbomb_player_id, tm_player_id, …)` e `player_xref_all(…)`
- `club_xref(tm_club_id, tm_club_name, alias)`

Viste principali
- EURO24: `transfers_euro24_window`, `v_players_transfers_euro24_window`, `v_players_transfers_euro24_window_all`.
- ALL (storico): `transfers_nt_window`, `v_players_transfers_nt_window`, `v_players_transfers_nt_window_all`.

## Schema ER

Per una visualizzazione immediata dello schema (tabelle principali, viste e relazioni) fai riferimento all'immagine SVG generata in questa cartella:

`[docs/SCHEMA_EURO24.svg]`


## Scelte “di prodotto” e compromessi

- Valuations fuori dal DB: manteniamo i time‑series dei market value esternamente per poter scegliere in seguito il cutoff giusto (pre‑EURO, pre‑transfer, ecc.) senza ricreare il DB.
- Due XRef separati: qualità vs copertura. Le viste “_all” sono utili quando serve massimizzare righe anche a rischio di qualche falso positivo.
- SQLite: semplicità e riproducibilità vincono sulla scalabilità (qui sufficiente). In caso di crescita, lo schema è facilmente migrabile ad un RDBMS maggiore.
- Minimo indispensabile: non inseriamo statistiche di club stagionali; il perimetro è “nazionali in tornei”. Questo rende il segnale coerente col target (fee post‑torneo) e limita confondenti.

## Esecuzione end‑to‑end (riproducibile)

Puoi eseguire tutto con lo script di progetto alla radice:

```bash
./run.sh
```

Oppure i singoli step:

```bash
# DB storici (nazionali dal 2018)
python -m src.pipeline.run_all_tournaments --all --db data/db/all_tournaments.sqlite

# DB EURO24
python -m src.pipeline.run_euro24 --all --db data/db/euro24.sqlite
```


## Considerazioni

- Normalizzazione club minima (alias di base); non modelliamo catene di trasferimenti o prestiti.
- Eventi completi (events_meta) sono inclusi per completezza ma non sempre usati in modeling.
- Le stats per‑90 non sono materializzate nel DB per evitare duplicazioni/ambiguità; vengono calcolate in fase di feature building.

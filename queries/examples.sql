-- Esempi rapidi sul DB EURO24
-- Seleziona queste righe e premi Cmd/Ctrl+Enter con SQLTools

-- 1) Controllo tabelle e conteggi
SELECT 'teams' AS tbl, COUNT(*) AS n FROM teams
UNION ALL SELECT 'players', COUNT(*) FROM players
UNION ALL SELECT 'matches', COUNT(*) FROM matches
UNION ALL SELECT 'player_match_stats', COUNT(*) FROM player_match_stats
UNION ALL SELECT 'player_appearances', COUNT(*) FROM player_appearances
UNION ALL SELECT 'events_meta', COUNT(*) FROM events_meta
UNION ALL SELECT 'referees', COUNT(*) FROM referees
UNION ALL SELECT 'venues', COUNT(*) FROM venues
UNION ALL SELECT 'transfers', COUNT(*) FROM transfers
UNION ALL SELECT 'player_xref', COUNT(*) FROM player_xref
UNION ALL SELECT 'club_xref', COUNT(*) FROM club_xref;

-- 2) Top marcatori
SELECT p.player_name, SUM(s.goals) AS goals
FROM player_match_stats s
JOIN players p ON p.player_id = s.player_id
GROUP BY p.player_name
ORDER BY goals DESC
LIMIT 10;

-- 3) Partite di una squadra (sostituisci 774 con l'ID desiderato)
SELECT match_id, match_date, home_team_name, away_team_name, home_score, away_score
FROM matches
WHERE home_team_id = 774 OR away_team_id = 774
ORDER BY match_date;


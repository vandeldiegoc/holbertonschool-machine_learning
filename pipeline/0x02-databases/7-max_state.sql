-- displays the max temperature of each state
SELECT DISTINCT state, MAX(value) as max_temp from temperatures GROUP BY state ORDER BY state;
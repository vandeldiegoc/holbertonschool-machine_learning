-- displays the average temperature by city ordered
SELECT city, AVG(value) as temp FROM temperatures GROUP BY city ORDER BY temp DESC;
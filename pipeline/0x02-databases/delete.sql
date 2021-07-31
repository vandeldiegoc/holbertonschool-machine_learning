WITH CTE AS(
   SELECT *,
       ROW_NUMBER()OVER(PARTITION BY name ORDER BY id) as rn
   FROM second_table
)
DELETE FROM CTE WHERE rn > 1
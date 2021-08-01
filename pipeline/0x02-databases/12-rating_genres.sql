--  script that lists all genres in the database
SELECT name, sum(tv_show_ratings.rate) AS rating
from tv_genres tv_g, tv_show_genres, tv_show_ratings
WHERE tv_show_genres.genre_id = tv_g.id 
AND tv_show_genres.show_id = tv_show_ratings.show_id
GROUP BY name
ORDER BY rating DESC;
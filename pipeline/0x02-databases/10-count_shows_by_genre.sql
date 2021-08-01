--  and displays the number of shows linked to each
SELECT tv_genres.name as genre, count(*) as number_of_shows 
FROM tv_genres, tv_show_genres, tv_shows
WHERE tv_genres.id=tv_show_genres.genre_id 
AND tv_shows.id=tv_show_genres.show_id 
GROUP BY tv_genres.name
HAVING number_of_shows != 0
ORDER BY number_of_shows DESC;
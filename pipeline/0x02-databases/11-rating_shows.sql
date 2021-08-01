-- lists all shows from hbtn_0d_tvshows_rate by their rating.
SELECT tv_shows.title, sum(tv_show_ratings.rate) 
from tv_shows, tv_show_ratings 
WHERE tv_shows.id = tv_show_ratings.show_id 
GROUP BY tv_shows.title
ORDER BY sum(tv_show_ratings.rate) DESC;
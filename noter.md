##### Hvad har vi lavet?

Malthe

- fik problemer med duplikater ifbm Hopkins nearest neighbour
- ca 4000 gengangere
- get_df() beholder første udgave af sangen, fjerner alle andre
- song recommendation
  - dropper "mode" fordi kun placeret i ekstremer
  - scaler ikke længere, men bruger min og max værdier
  - silhouette scores for de forskellige clusters
  - Vi skal se på hvordan vi bedst laver clustering
  - kan vi bruge feature engineering til at skabe moods, som vi kan bruge til at lave clusters
  - Maja: vi kan prøve at encode playlist_genre til som feature, når vi laver clusters
  - mangler at prøve flere clustering modeller

Simon

- Måske prøve PCA uden "mode"
- har lavet rythm_related udvalg af features, men model performer dårligere for R2 end med all ordinal features ift at predicte danceability
- danceability er korreleret med energy -
-

TODO

- find ud af datetime format bug (utils.py, ISO...)
-

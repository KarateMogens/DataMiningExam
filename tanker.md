##### resultater

- decision tree virker til at performe virkeligt godt - det er nok noget lignende spotify selv bruger til at klassificere genre
- regression på musical features er elendig, men nok også forventeligt - ellers kunne man måske bare lave et hit ved en simpel blanding af features (120bpm + Cdur + max loudness = monsterhit)

##### plots

- især EDM skiller sig ud i boxplots by genre (nok forventeligt)
- i feature by year virker aææe features til at have markant større udsving inden midt 60'erne, hvilket giver mening ift optagelsesudstyr, professionelle producere og lyd-engineers først kommer heromkiring
- loudness wars (https://en.wikipedia.org/wiki/Loudness_war) er helt klart en ting fra man begynder at have cd'er i 90'erne. Måske kan man også se indflydelsen fra Phil Spectors "wall of sound" i slut 60'erne? (https://en.wikipedia.org/wiki/Wall_of_Sound)
- speechiness udvikling giver mening ift at rap bliver mainstream fra slut 80'erne
- interessant at valence falder lige omkring finanskrisen
- duration falder kort efter at spotify bliver lanceret (2008)

##### features

- det giver mening at tempo har stor betydning - jeg forestiller mig at 99.9% af alt edm har en bpm på precis 120

- heller ikke overraskende at speechiness i hvert fald kan predicte rap
- giver også mening at mode og key har ret lav betydning - faktisk overraskende meget

##### genre

- det ville være sjovt at se om en clustering algoritme kommer op med de samme clusters som genre labels (både ift antal hvis vi bruger elbow method, men også om de clusters den komme frem til stemmer overens med de officialle genrebetegnelser)
- prøv også at predicte på playlist_subgenre

##### andet

- kig evt på sang titel ift playlist eller genre (naive bayes + TFIDF vector)
- hvilke andre features kan bruges ift at forudsige

##### spørgsmål

- skal vi lave et holdout set?
-

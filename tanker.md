##### resultater

- decision tree virker til at performe virkeligt godt - det er nok noget lignende spotify selv bruger til at klassificere genre
- regression på musical features er elendig, men nok også forventeligt - ellers kunne man måske bare lave et hit ved en simpel blanding af features (120bpm + Cdur + max loudness = monsterhit)

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

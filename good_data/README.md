# Let's keep this simple
"/processed/"   : where to save data processed from "/processed/" of Mr.Zhu (in "/preprocessing-zhu/mimic-iii/") to further fit our intentions.
"/incomplete/"  : where to save raw versions of train/val/test which mainly serve the purpose of precompute
"/curated/"     : where to save embeddings, entities... that we extracted, processed from PrimeKG, "/processed/",... to serve our purpose later on.
Another reason: to precompute some data so we won't have to rerun each experiment
"/complete/"    : where to save versions of train/val/test set we created using previous two. They are the final products, the end-goal.


"/processed/" -> "/incomplete/" -> "/curated/" -> "/complete/"
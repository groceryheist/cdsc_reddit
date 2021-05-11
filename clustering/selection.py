import fire
from select_affinity import select_affinity_clustering
from select_kmeans import select_kmeans_clustering

if __name__ == "__main__":
    fire.Fire({"kmeans":select_kmeans_clustering,
               "affinity":select_affinity_clustering})

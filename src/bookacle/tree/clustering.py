from collections import defaultdict
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import umap
from bookacle.tree.structures import Node
from sklearn.mixture import GaussianMixture


class ClustererLike(Protocol):
    def cluster(
        self, embeddings: npt.NDArray[np.float64], n_clusters: int | None = None
    ) -> list[npt.NDArray[np.int64]]: ...


class Clustering(Protocol):
    def perform_clustering(
        self,
        nodes: list[Node],
        embedding_model_name: str,
        tokenizer: Any,
        max_length_in_cluster: int = 3500,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
    ) -> list[list[Node]]: ...


def umap_reduce_embeddings(
    embeddings: npt.NDArray[np.float64],
    n_components: int,
    neighbors: int = 10,
    metric: str = "cosine",
) -> npt.NDArray[np.float64]:
    reduction = umap.UMAP(
        n_neighbors=neighbors,
        n_components=n_components,
        metric=metric,
    ).fit_transform(embeddings)

    assert isinstance(reduction, np.ndarray)

    return reduction


class GMMClusterer:
    def __init__(
        self,
        reduction_dim: int,
        max_clusters: int = 50,
        threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.reduction_dim = reduction_dim
        self.max_clusters = max_clusters
        self.threshold = threshold
        self.random_state = random_state

    def get_optimal_clusters_count(self, embeddings: npt.NDArray[np.float64]) -> int:
        max_clusters = min(self.max_clusters, embeddings.shape[0])
        n_clusters = np.arange(1, max_clusters)

        bics = np.array(
            [
                GaussianMixture(n_components=n, random_state=self.random_state)
                .fit(embeddings)
                .bic(embeddings)
                for n in n_clusters
            ]
        )

        return n_clusters[np.argmin(bics)]

    def get_clusters(
        self, embeddings: npt.NDArray[np.float64], n_clusters: int | None = None
    ) -> tuple[int, npt.NDArray[np.int64]]:
        if n_clusters is None:
            n_clusters = self.get_optimal_clusters_count(embeddings)

        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        gm = gm.fit(embeddings)

        probabilities = gm.predict_proba(embeddings)

        _, clusters = np.asarray(probabilities > self.threshold).nonzero()

        return n_clusters, clusters

    def reduce_and_cluster_embeddings(
        self,
        embeddings: npt.NDArray[np.float64],
        n_components: int,
        n_neighbors: int,
        n_clusters: int | None = None,
    ) -> tuple[int, npt.NDArray[np.int64]]:
        reduced_embeddings_global = umap_reduce_embeddings(
            embeddings=embeddings,
            n_components=n_components,
            neighbors=n_neighbors,
        )

        return self.get_clusters(
            embeddings=reduced_embeddings_global, n_clusters=n_clusters
        )

    def cluster_locally(
        self,
        embeddings: npt.NDArray[np.float64],
        cluster_embeddings_global: npt.NDArray[np.float64],
        n_clusters: int | None = None,
        n_neighbors: int = 10,
    ) -> tuple[int, dict[int, npt.NDArray[np.int64]]]:
        n_samples = cluster_embeddings_global.shape[0]

        if n_samples <= self.reduction_dim + 1:
            local_clusters = np.zeros(n_samples, dtype=np.int64)
            n_local_clusters = 1
        else:
            n_local_clusters, local_clusters = self.reduce_and_cluster_embeddings(
                embeddings=cluster_embeddings_global,
                n_components=self.reduction_dim,
                n_neighbors=n_neighbors,
                n_clusters=n_clusters,
            )

        clusters: dict[int, npt.NDArray[np.int64]] = {}

        for local_cluster_index in range(n_local_clusters):
            cluster_embeddings_local = cluster_embeddings_global[
                local_clusters == local_cluster_index
            ]

            # Use broadcasting to find the indices where embeddings match
            clusters[local_cluster_index] = np.where(
                (embeddings == cluster_embeddings_local[:, None]).all(axis=2)
            )[1]

        return n_local_clusters, clusters

    def cluster(
        self,
        embeddings: npt.NDArray[np.float64],
        n_neighbors_global: int | None = None,
        n_neighbors_local: int = 10,
        n_clusters_global: int | None = None,
        n_clusters_local: int | None = None,
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        if n_neighbors_global is None:
            n_neighbors_global = int((len(embeddings) - 1) ** 0.5)

        n_clusters_global, clusters = self.reduce_and_cluster_embeddings(
            embeddings=embeddings,
            n_components=min(self.reduction_dim, embeddings.shape[0] - 2),
            n_neighbors=n_neighbors_global,
            n_clusters=n_clusters_global,
        )

        node_to_cluster: dict[int, list[int]] = defaultdict(list)
        cluster_to_node: dict[int, list[int]] = defaultdict(list)
        total_clusters = 0

        for cluster_index in range(n_clusters_global):
            cluster_embeddings_global = embeddings[clusters == cluster_index]

            n_samples = cluster_embeddings_global.shape[0]

            if n_samples == 0:
                continue

            n_local_clusters, local_clusters = self.cluster_locally(
                embeddings=embeddings,
                cluster_embeddings_global=cluster_embeddings_global,
                n_clusters=n_clusters_local,
                n_neighbors=n_neighbors_local,
            )

            for local_cluster_index, indices in local_clusters.items():
                for idx in indices:
                    node_to_cluster[idx].append(total_clusters + local_cluster_index)
                    cluster_to_node[total_clusters + local_cluster_index].append(idx)

            total_clusters += n_local_clusters

        return node_to_cluster, cluster_to_node


if __name__ == "__main__":
    from bookacle.models import HuggingFaceEmbeddingModel
    from bookacle.splitter import HuggingFaceDocumentSplitter
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader("data/c-language.pdf")
    documents = loader.load()
    documents = documents[:100]

    embedding_model = HuggingFaceEmbeddingModel(
        model_name="sentence-transformers/paraphrase-albert-small-v2", use_gpu=True
    )
    document_splitter = HuggingFaceDocumentSplitter(tokenizer=embedding_model.tokenizer)

    chunks = document_splitter(documents=documents)
    embeddings = embedding_model.model.embed_documents(
        texts=[chunk.page_content for chunk in chunks]
    )

    gmm_clusterer = GMMClusterer(reduction_dim=10)
    node_to_cluster, cluster_to_node = gmm_clusterer.cluster(
        embeddings=np.array(embeddings)
    )

    print(node_to_cluster)
    print(cluster_to_node)

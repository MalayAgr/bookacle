from collections import defaultdict
from typing import Protocol

import numpy as np
import numpy.typing as npt
import umap
from bookacle.tokenizer import TokenizerLike
from bookacle.tree.structures import Node
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture


def umap_reduce_embeddings(
    embeddings: npt.NDArray[np.float64],
    n_components: int,
    neighbors: int = 10,
    metric: str = "cosine",
    low_memory: bool = True,
) -> npt.NDArray[np.float64]:
    reduction = umap.UMAP(
        n_neighbors=neighbors,
        n_components=n_components,
        metric=metric,
        low_memory=low_memory,
    ).fit_transform(embeddings)

    assert isinstance(reduction, np.ndarray)

    return reduction


class ClusteringBackendLike(Protocol):
    def cluster(
        self,
        embeddings: npt.NDArray[np.float64],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]: ...


class ClusteringFunctionLike(Protocol):
    def __call__(  # type: ignore
        self,
        nodes: list[Node],
        tokenizer: TokenizerLike,
        clustering_backend: ClusteringBackendLike | None = None,
        max_length_in_cluster: int = 3500,
        reduction_dimension: int = 10,
        *args,
        **kwargs,
    ) -> list[list[Node]]: ...


class GMMClusteringBackend:
    def __init__(
        self,
        reduction_dim: int,
        max_clusters: int = 50,
        random_state: int = 42,
        n_neighbors_global: int | None = None,
        n_neighbors_local: int = 10,
        n_clusters_global: int | None = None,
        n_clusters_local: int | None = None,
        umap_metric: str = "cosine",
        umap_low_memory: bool = True,
    ) -> None:
        self.reduction_dim = reduction_dim
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.n_neighbors_global = n_neighbors_global
        self.n_neighbors_local = n_neighbors_local
        self.n_clusters_global = n_clusters_global
        self.n_clusters_local = n_clusters_local
        self.umap_metric = umap_metric
        self.umap_low_memory = umap_low_memory

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(reduction_dim={self.reduction_dim}, "
            f"max_clusters={self.max_clusters}, random_state={self.random_state}, "
            f"n_neighbors_global={self.n_neighbors_global}, n_neighbors_local={self.n_neighbors_local}, "
            f"n_clusters_global={self.n_clusters_global}, n_clusters_local={self.n_clusters_local})"
        )

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

        clusters = gm.predict(embeddings)

        return n_clusters, clusters

    def reduce_and_cluster_embeddings(
        self,
        embeddings: npt.NDArray[np.float64],
        n_components: int,
        n_neighbors: int,
        n_clusters: int | None = None,
    ) -> tuple[int, npt.NDArray[np.int64]]:
        reduced_embeddings = umap_reduce_embeddings(
            embeddings=embeddings,
            n_components=n_components,
            neighbors=n_neighbors,
            metric=self.umap_metric,
            low_memory=self.umap_low_memory,
        )

        return self.get_clusters(embeddings=reduced_embeddings, n_clusters=n_clusters)

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

    def process_single_cluster(
        self,
        cluster_index: int,
        embeddings: npt.NDArray[np.float64],
        clusters: npt.NDArray[np.int64],
    ) -> tuple[int, dict[int, npt.NDArray[np.int64]]] | None:
        cluster_embeddings_global = embeddings[clusters == cluster_index]
        n_samples = cluster_embeddings_global.shape[0]

        if n_samples == 0:
            return None  # Skip empty clusters

        n_local_clusters, local_clusters = self.cluster_locally(
            embeddings=embeddings,
            cluster_embeddings_global=cluster_embeddings_global,
            n_clusters=self.n_clusters_local,
            n_neighbors=self.n_neighbors_local,
        )

        return n_local_clusters, local_clusters

    def cluster(
        self,
        embeddings: npt.NDArray[np.float64],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        n_neighbors_global = (
            int((len(embeddings) - 1) ** 0.5)
            if self.n_neighbors_global is None
            else self.n_neighbors_global
        )

        n_clusters_global, clusters = self.reduce_and_cluster_embeddings(
            embeddings=embeddings,
            n_components=min(self.reduction_dim, embeddings.shape[0] - 2),
            n_neighbors=n_neighbors_global,
            n_clusters=self.n_clusters_global,
        )

        node_to_cluster: dict[int, list[int]] = defaultdict(list)
        cluster_to_node: dict[int, list[int]] = defaultdict(list)
        total_clusters: int = 0

        results = Parallel(n_jobs=-1)(
            delayed(self.process_single_cluster)(cluster_index, embeddings, clusters)
            for cluster_index in range(n_clusters_global)
        )

        for result in results:
            if result is None:
                continue

            n_local_clusters, local_clusters = result
            for local_cluster_index, indices in local_clusters.items():
                for idx in indices:
                    node_to_cluster[idx].append(total_clusters + local_cluster_index)
                    cluster_to_node[total_clusters + local_cluster_index].append(idx)

            total_clusters += n_local_clusters

        return node_to_cluster, cluster_to_node


def raptor_clustering(
    nodes: list[Node],
    tokenizer: TokenizerLike,
    clustering_backend: ClusteringBackendLike | None = None,
    max_length_in_cluster: int = 3500,
    reduction_dimension: int = 10,
) -> list[list[Node]]:
    if clustering_backend is None:
        clustering_backend = GMMClusteringBackend(reduction_dim=reduction_dimension)

    embeddings = np.array([node.embeddings for node in nodes])

    _, cluster_to_node = clustering_backend.cluster(embeddings=embeddings)

    node_clusters: list[list[Node]] = []

    for _, node_indices in cluster_to_node.items():
        cluster_nodes = [nodes[idx] for idx in node_indices]
        total_length = sum(len(tokenizer.encode(node.text)) for node in cluster_nodes)

        if len(cluster_nodes) == 1 or total_length <= max_length_in_cluster:
            node_clusters.append(cluster_nodes)
            continue

        reclustered_nodes = raptor_clustering(
            nodes=cluster_nodes,
            tokenizer=tokenizer,
            clustering_backend=clustering_backend,
            max_length_in_cluster=max_length_in_cluster,
            reduction_dimension=reduction_dimension,
        )

        node_clusters.extend(reclustered_nodes)

    return node_clusters


if __name__ == "__main__":
    from bookacle.models.embedding import SentenceTransformerEmbeddingModel
    from bookacle.splitters import HuggingFaceTextSplitter
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader("data/c-language.pdf")
    documents = loader.load()
    documents = documents[:20]

    embedding_model = SentenceTransformerEmbeddingModel(
        model_name="sentence-transformers/paraphrase-albert-small-v2", use_gpu=True
    )
    document_splitter = HuggingFaceTextSplitter(tokenizer=embedding_model.tokenizer)

    chunks = document_splitter(documents=documents)
    embeddings = embedding_model.embed(text=[chunk.page_content for chunk in chunks])

    gmm_clusterer = GMMClusteringBackend(reduction_dim=10)

    clusters = gmm_clusterer.cluster(embeddings=np.array(embeddings))

    print(clusters)

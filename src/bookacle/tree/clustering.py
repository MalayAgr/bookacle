"""Clustering module for the tree structure."""

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
    """Reduce the dimensionality of the embeddings using UMAP.

    Args:
        embeddings: The embeddings to reduce.
        n_components: The number of components in the reduced space.
        neighbors: The number of neighbors to use for UMAP.
        metric: The metric to use for UMAP.
        low_memory: Whether to use low memory mode for UMAP.

    Returns:
        The reduced embeddings.
    """
    reduction = umap.UMAP(
        n_neighbors=neighbors,
        n_components=n_components,
        metric=metric,
        low_memory=low_memory,
    ).fit_transform(embeddings)

    assert isinstance(reduction, np.ndarray)

    return reduction


class ClusteringBackendLike(Protocol):
    """A protocol that defines the interface a clustering backend should implement."""

    def cluster(  # type: ignore
        self, embeddings: npt.NDArray[np.float64], *args, **kwargs
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Cluster the embeddings.

        Args:
            embeddings: The embeddings to cluster.
            *args: Additional arguments to pass to the clustering function.
            **kwargs: Additional keyword arguments to pass to the clustering function.

        Returns:
            A tuple containing the mapping of embeddings to clusters and the mapping of clusters to embeddings.
        """
        ...


class ClusteringFunctionLike(Protocol):
    """A protocol that defines the interface a clustering function should implement."""

    def __call__(  # type: ignore
        self,
        nodes: list[Node],
        tokenizer: TokenizerLike,
        clustering_backend: ClusteringBackendLike | None = None,
        max_length_in_cluster: int = 3500,
        reduction_dimension: int = 10,
        *args,
        **kwargs,
    ) -> list[list[Node]]:
        """Cluster nodes using the given clustering backend.

        Args:
            nodes: The nodes to cluster.
            tokenizer: The tokenizer to use to calculate the total length of text in each cluster.
            clustering_backend: The clustering backend to use.
            max_length_in_cluster: The maximum length of text in a cluster.
            reduction_dimension: The dimension to reduce the embeddings to before clustering.
            *args: Additional arguments to pass to the clustering function.
            **kwargs: Additional keyword arguments to pass to the clustering function.

        Returns:
            The clustered nodes.
        """
        ...


class GMMClusteringBackend:
    """A Gaussian Mixture Model (GMM) clustering backend.

    It implements the [ClusteringBackendLike][bookacle.tree.clustering.ClusteringBackendLike] protocol.

    Attributes:
        reduction_dim (int): The dimension to reduce the embeddings to before clustering.
        max_clusters (int): The maximum number of clusters to use.
        random_state (int): Random state for reproducibility.
        n_neighbors_global (int | None): The number of neighbors to use for global clustering.
        n_neighbors_local (int | None): The number of neighbors to use for local clustering.
        n_clusters_global (int | None): The number of clusters to use for global clustering.
        n_clusters_local (int | None): The number of clusters to use for local clustering.
        umap_metric (str): The metric to use for UMAP.
        umap_low_memory (bool): Whether to use low memory mode for UMAP.
    """

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
        """Initialize the GMM clustering backend.

        Args:
            reduction_dim: The dimension to reduce the embeddings to before clustering.
            max_clusters: The maximum number of clusters to use.
            random_state: Random state for reproducibility.
            n_neighbors_global: The number of neighbors to use for global clustering.
                                When `None`, it is calculated using Bayesian Information Criterion (BIC).
            n_neighbors_local: The number of neighbors to use for local clustering.
                                When `None`, it is calculated using Bayesian Information Criterion (BIC).
            n_clusters_global: The number of clusters to use for global clustering.
            n_clusters_local: The number of clusters to use for local clustering.
            umap_metric: The metric to use for UMAP.
            umap_low_memory: Whether to use low memory mode for UMAP.
        """
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
        """Get the optimal number of clusters using the Bayesian Information Criterion (BIC).

        The method fits multiple Gaussian Mixture Models (GMM) to the embeddings
        and calculates the BIC for each. The number of clusters with the lowest BIC
        score is selected as the optimal number.

        Args:
            embeddings: The embeddings to cluster.

        Returns:
            The optimal number of clusters.
        """
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
        self, embeddings: npt.NDArray[np.float64], n_clusters: int
    ) -> npt.NDArray[np.int64]:
        """Fit the GMM model to embeddings and return cluster assignments.

        Args:
            embeddings: The embeddings to cluster.
            n_clusters: The number of clusters to use.

        Returns:
            The cluster assignments.
        """
        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        gm = gm.fit(embeddings)

        clusters = gm.predict(embeddings)

        return clusters

    def reduce_and_cluster_embeddings(
        self,
        embeddings: npt.NDArray[np.float64],
        n_components: int,
        n_neighbors: int,
        n_clusters: int | None = None,
    ) -> npt.NDArray[np.int64]:
        """Reduce the dimensionality of the embeddings using UMAP and cluster them using GMM.

        It uses [umap_reduce_embeddings()][bookacle.tree.clustering.umap_reduce_embeddings] to reduce the dimensionality
        of the embeddings and [get_clusters()][bookacle.tree.clustering.GMMClusteringBackend.get_clusters] to cluster them.

        Generally speaking, this method should be used instead of calling
        [get_clusters()][bookacle.tree.clustering.GMMClusteringBackend.get_clusters] directly.

        Args:
            embeddings: The embeddings to cluster.
            n_components: The number of components in the reduced space.
            n_neighbors: The number of neighbors to use for UMAP.
            n_clusters: The number of clusters to use. When `None`, the optimal number of clusters is
                        calculated using
                        [get_optimal_clusters_count()][bookacle.tree.clustering.GMMClusteringBackend.get_optimal_clusters_count].

        Returns:
            The cluster assignments.
        """
        reduced_embeddings = umap_reduce_embeddings(
            embeddings=embeddings,
            n_components=n_components,
            neighbors=n_neighbors,
            metric=self.umap_metric,
            low_memory=self.umap_low_memory,
        )

        if n_clusters is None:
            n_clusters = self.get_optimal_clusters_count(embeddings=reduced_embeddings)

        return self.get_clusters(embeddings=reduced_embeddings, n_clusters=n_clusters)

    def cluster_locally(
        self,
        embeddings: npt.NDArray[np.float64],
        global_cluster_indices: npt.NDArray[np.int64],
        n_clusters: int | None = None,
        n_neighbors: int = 10,
    ) -> list[npt.NDArray[np.int64]]:
        """Cluster the embeddings of a global cluster locally.
        In other words, create new clusters from the embeddings of a global cluster.

        If the number of embeddings in the global cluster is less than or equal to
        `reduction_dim`, the global cluster is returned as is in a singleton list.

        Args:
            embeddings: The overall embeddings.
            global_cluster_indices: The indices of the embeddings belonging to the global cluster.
            n_clusters: The number of clusters to output.
                        When `None`, the optimal number of clusters is calculated using BIC.
            n_neighbors: The number of neighbors to use for UMAP.

        Returns:
            The local clusters.

        Examples:
            ```python exec="true" source="above" result="python"
            import numpy as np
            from bookacle.tree.clustering import GMMClusteringBackend

            backend = GMMClusteringBackend(reduction_dim=10)
            embeddings = np.random.rand(100, 768)
            indices = np.random.choice(100, 30)
            clusters = backend.cluster_locally(
                embeddings=embeddings,
                global_cluster_indices=indices,
                n_clusters=5,
                n_neighbors=10,
            )
            print(clusters)
            ```
        """
        if global_cluster_indices.shape[0] <= self.reduction_dim + 1:
            return [global_cluster_indices]

        local_clusters = self.reduce_and_cluster_embeddings(
            embeddings=embeddings[global_cluster_indices],
            n_components=self.reduction_dim,
            n_neighbors=n_neighbors,
            n_clusters=n_clusters,
        )

        return [
            global_cluster_indices[local_clusters == local_cluster_idx]
            for local_cluster_idx in np.unique(local_clusters)
        ]

    def _process_single_cluster(
        self,
        cluster_index: int,
        embeddings: npt.NDArray[np.float64],
        clusters: npt.NDArray[np.int64],
    ) -> list[npt.NDArray[np.int64]]:
        """Process a single global cluster.

        Here, process implies checking if the cluster is empty and clustering the embeddings of the cluster locally.

        Args:
            cluster_index: The index of the global cluster.
            embeddings: The overall embeddings.
            clusters: The cluster assignments for each embedding.

        Returns:
            The local clusters.
        """
        cluster_mask = clusters == cluster_index
        indices = np.where(cluster_mask)[0]

        if indices.shape[0] == 0:
            return []  # Skip empty clusters

        return self.cluster_locally(
            embeddings=embeddings,
            global_cluster_indices=indices,
            n_clusters=self.n_clusters_local,
            n_neighbors=self.n_neighbors_local,
        )

    def cluster(
        self, embeddings: npt.NDArray[np.float64]
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Cluster the embeddings.

        The clustering is done as follows:
            - Global clustering: The embeddings are reduced and clustered globally.
                                 That is, the entire set of embeddings is clustered.
            - Local clustering: The embeddings of each global cluster are reduced and clustered.
            - At the end, all local clusters are aggregated into a single result.

        [Parallel][joblib.Parallel] from `joblib` is used to parallelize the clustering of each global cluster.

        Args:
            embeddings: The embeddings to cluster.

        Returns:
            A mapping of embeddings to clusters
            A mapping of clusters to embeddings.

        Examples:
            ```python exec="true" source="material-block" result="python"
            import numpy as np
            from bookacle.tree.clustering import GMMClusteringBackend

            backend = GMMClusteringBackend(reduction_dim=10, n_clusters_global=3)
            embeddings = np.random.rand(10, 768)
            emb_to_clusters, clusters_to_emb = backend.cluster(embeddings=embeddings)
            print(clusters_to_emb)
            ```
        """
        n_neighbors_global = (
            int((len(embeddings) - 1) ** 0.5)
            if self.n_neighbors_global is None
            else self.n_neighbors_global
        )

        clusters = self.reduce_and_cluster_embeddings(
            embeddings=embeddings,
            n_components=min(self.reduction_dim, embeddings.shape[0] - 2),
            n_neighbors=n_neighbors_global,
            n_clusters=self.n_clusters_global,
        )

        embedding_to_cluster: dict[int, list[int]] = defaultdict(list)
        cluster_to_embedding: dict[int, list[int]] = defaultdict(list)
        total_clusters: int = 0

        all_clusters: list[list[npt.NDArray[np.int64]]] = Parallel(n_jobs=-1)(  # type: ignore
            delayed(self._process_single_cluster)(cluster_index, embeddings, clusters)
            for cluster_index in np.unique(clusters)
        )

        for cluster in all_clusters:
            if not cluster:
                continue

            for local_cluster_index, indices in enumerate(cluster):
                global_cluster_idx = total_clusters + local_cluster_index
                cluster_to_embedding[global_cluster_idx] = indices.tolist()

                for idx in indices:
                    embedding_to_cluster[idx].append(global_cluster_idx)

            total_clusters += len(cluster)

        return embedding_to_cluster, cluster_to_embedding


def raptor_clustering(
    nodes: list[Node],
    tokenizer: TokenizerLike,
    clustering_backend: ClusteringBackendLike | None = None,
    max_length_in_cluster: int = 3500,
    reduction_dimension: int = 10,
) -> list[list[Node]]:
    """Cluster nodes using RAPTOR clustering.

    It implements the [ClusteringFunctionLike][bookacle.tree.clustering.ClusteringFunctionLike] protocol.

    To cluster the nodes:
        - The nodes are clustered using the given clustering backend.
        - For each cluster:
            - If the cluster has only one node or the total length of text in the
            cluster is less than the maximum length, the cluster is kept as is.
            - Otherwise, the cluster is recursively clustered.
    """
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
    from bookacle.loaders import pymupdf_loader
    from bookacle.models.embedding import SentenceTransformerEmbeddingModel
    from bookacle.splitters import HuggingFaceTextSplitter

    documents = pymupdf_loader("./data/c-language.pdf", start_page=0, end_page=10)

    embedding_model = SentenceTransformerEmbeddingModel(
        model_name="sentence-transformers/paraphrase-albert-small-v2", use_gpu=True
    )
    document_splitter = HuggingFaceTextSplitter(tokenizer=embedding_model.tokenizer)

    chunks = document_splitter(documents=documents)
    embeddings = embedding_model.embed(text=[chunk["page_content"] for chunk in chunks])

    gmm_clusterer = GMMClusteringBackend(reduction_dim=10)

    clusters = gmm_clusterer.cluster(embeddings=np.array(embeddings))

    print(clusters)

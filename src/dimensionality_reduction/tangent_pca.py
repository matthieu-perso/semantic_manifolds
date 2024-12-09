import numpy as np
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from sklearn.metrics import mean_squared_error, explained_variance_score, pairwise_distances
from sklearn.manifold import trustworthiness

class TangentPCAEvaluator:
    def __init__(self, data, reduced_data, log_data, approx_data, pca):
        self.data = data
        self.reduced_data = reduced_data
        self.log_data = log_data
        self.approx_data = approx_data
        self.pca = pca

    def evaluate(self):
        self._evaluate_variance()
        self._evaluate_reconstruction_error()
        self._evaluate_trustworthiness()
        self._evaluate_global_structure()

    def _evaluate_variance(self):
        explained_variance_forward = sum(self.pca.explained_variance_ratio_)
        explained_variance_backward = sum(self.pca.inverse_transform(self.pca.transform(self.log_data)).var(axis=0) / self.log_data.var(axis=0))
        print(f"Explained Variance (Forward): {explained_variance_forward:.4f}")

    def _evaluate_reconstruction_error(self):
        reconstruction_error_fro = np.linalg.norm(self.log_data - self.pca.inverse_transform(self.pca.transform(self.log_data)), 'fro')
        print(f"Reconstruction Error (Frobenius norm): {reconstruction_error_fro:.4f}")

    def _evaluate_trustworthiness(self):
        trustworthiness_score = trustworthiness(self.data, self.reduced_data, n_neighbors=5)
        print(f"Trustworthiness: {trustworthiness_score:.4f}")

    def _evaluate_global_structure(self):
        original_distances = pairwise_distances(self.data)
        reduced_distances = pairwise_distances(self.reduced_data)
        correlation = np.corrcoef(original_distances.ravel(), reduced_distances.ravel())[0, 1]
        print(f"Correlation between original and reduced distances: {correlation:.4f}")
        continuity_score = trustworthiness(self.reduced_data, self.data, n_neighbors=5)
        print(f"Continuity: {continuity_score:.4f}")
        explained_variance = explained_variance_score(self.log_data, self.pca.inverse_transform(self.pca.transform(self.log_data)))
        print(f"Explained Variance Score: {explained_variance:.4f}")

class TangentPCAProcessor:
    def __init__(self, embeddings_raw, n_components=400):
        self.embeddings_raw = embeddings_raw
        self.n_components = n_components
        self.sphere = None
        self.frechet_mean = None
        self.log_data = None
        self.reduced_data = None
        self.approx_data = None
        self.pca = None

    def process(self):
        self._define_manifold()
        self._compute_frechet_mean()
        self._map_to_tangent_space()
        self._perform_pca()
        self._map_back_to_sphere()
        return self.log_data, self.reduced_data, self.approx_data, self.pca

    def _define_manifold(self):
        dim = len(self.embeddings_raw[0])
        self.sphere = Hypersphere(dim)
        self.data = gs.array(self.embeddings_raw)

    def _compute_frechet_mean(self):
        mean_estimator = FrechetMean(self.sphere)
        self.frechet_mean = mean_estimator.fit(self.data).estimate_

    def _map_to_tangent_space(self):
        self.log_data = self.sphere.metric.log(self.data, base_point=self.frechet_mean)

    def _perform_pca(self):
        self.pca = TangentPCA(self.sphere, n_components=self.n_components)
        self.pca.fit(self.log_data, base_point=self.frechet_mean)
        self.reduced_data = self.pca.transform(self.log_data)

    def _map_back_to_sphere(self):
        inverse_transformed_data = self.pca.inverse_transform(self.reduced_data)
        self.approx_data = self.sphere.metric.exp(inverse_transformed_data, base_point=self.frechet_mean)

if __name__ == "__main__":

    # just for testing
    embeddings_raw = np.random.rand(100, 300)  
    processor = TangentPCAProcessor(embeddings_raw)
    log_data, reduced_data, approx_data, pca = processor.process()

    evaluator = TangentPCAEvaluator(processor.data, reduced_data, log_data, approx_data, pca)
    evaluator.evaluate()

    # map back to sphere
    inverse_transformed_data = pca.inverse_transform(reduced_data)
    approx_data = processor.sphere.metric.exp(inverse_transformed_data, base_point=processor.frechet_mean)
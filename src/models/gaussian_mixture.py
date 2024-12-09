import optuna
import logging
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import silhouette_score, log_loss, mean_squared_error
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def objective(trial, data):
    n_components = trial.suggest_int("n_components", 2, 10)
    covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
    reg_covar = trial.suggest_loguniform("reg_covar", 1e-6, 1e-3)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=42
    )
    gmm.fit(data)
    return -gmm.bic(data)

def optimize_gmm(data, n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, data), n_trials=n_trials)
    print("Best Parameters:", study.best_params)
    print("Best BIC:", study.best_value)
    return study

def cross_validate_gmm(gmm, data, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    log_likelihoods, bics = [], []
    for train_idx, test_idx in skf.split(data, np.zeros(data.shape[0])):
        X_train, X_test = data[train_idx], data[test_idx]
        gmm.fit(X_train)
        log_likelihoods.append(gmm.score(X_test) * len(X_test))
        bics.append(gmm.bic(X_test))
    return {
        "mean_log_likelihood": np.mean(log_likelihoods),
        "std_log_likelihood": np.std(log_likelihoods),
        "mean_bic": np.mean(bics),
        "std_bic": np.std(bics)
    }

def residual_analysis(gmm, data):
    log_likelihoods = gmm.score_samples(data)
    print(f"\nLog-Likelihood Stats:")
    print(f"Mean: {np.mean(log_likelihoods):.2f}, Std: {np.std(log_likelihoods):.2f}")
    return log_likelihoods


def robustness_check(gmm, data, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cluster_counts = []
    for train_idx, test_idx in skf.split(data, np.zeros(data.shape[0])):
        X_train = data[train_idx]
        gmm.fit(X_train)
        cluster_counts.append(len(np.unique(gmm.predict(X_train))))
    print(f"\nCluster Stability Check: Cluster Counts Across Splits: {cluster_counts}")

def predict_points(gmm, X):
    labels = gmm.predict(X)
    return gmm.means_[labels]

def perplexity(log_likelihood, n_points):
    return np.exp(-log_likelihood / n_points)


def run_gmm_model(reduced_data):
    X_train, X_test = train_test_split(reduced_data, test_size=0.2, random_state=42)

    study = optimize_gmm(reduced_data)
    best_params = study.best_params
    best_gmm = GaussianMixture(
        n_components=best_params['n_components'],
        covariance_type=best_params['covariance_type'],
        reg_covar=best_params['reg_covar'],
        random_state=42
    )
    best_gmm.fit(X_train)

    train_log_likelihood = best_gmm.score(X_train) * len(X_train)
    train_bic = best_gmm.bic(X_train)
    train_aic = best_gmm.aic(X_train)
    test_log_likelihood = best_gmm.score(X_test) * len(X_test)
    test_bic = best_gmm.bic(X_test)
    test_aic = best_gmm.aic(X_test)

    X_train_reconstructed = predict_points(best_gmm, X_train)
    train_mse = mean_squared_error(X_train, X_train_reconstructed)
    X_test_reconstructed = predict_points(best_gmm, X_test)
    test_mse = mean_squared_error(X_test, X_test_reconstructed)

    train_perplexity = perplexity(train_log_likelihood, len(X_train))
    test_perplexity = perplexity(test_log_likelihood, len(X_test))

    logging.info("--- Model Validation ---")
    logging.info(f"Train Log-Likelihood: {train_log_likelihood:.2f}")
    logging.info(f"Train BIC: {train_bic:.2f}")
    logging.info(f"Train AIC: {train_aic:.2f}")
    logging.info(f"Train MSE (Reconstruction): {train_mse:.6f}")
    logging.info(f"Train Perplexity: {train_perplexity:.6f}")
    logging.info(f"Test Log-Likelihood: {test_log_likelihood:.2f}")
    logging.info(f"Test BIC: {test_bic:.2f}")
    logging.info(f"Test AIC: {test_aic:.2f}")
    logging.info(f"Test MSE (Reconstruction): {test_mse:.6f}")
    logging.info(f"Test Perplexity: {test_perplexity:.6f}")

    cv_results = cross_validate_gmm(best_gmm, reduced_data)
    logging.info(f"Cross-Validation Results:")
    logging.info(f"Mean Log-Likelihood: {cv_results['mean_log_likelihood']:.2f} ± {cv_results['std_log_likelihood']:.2f}")
    logging.info(f"Mean BIC: {cv_results['mean_bic']:.2f} ± {cv_results['std_bic']:.2f}")

    if best_params['n_components'] > 1:
        labels = best_gmm.fit_predict(reduced_data)
        silhouette = silhouette_score(reduced_data, labels)
        logging.info(f"Silhouette Score: {silhouette:.2f}")

    log_likelihoods = residual_analysis(best_gmm, reduced_data)
    robustness_check(best_gmm, reduced_data)


def classify_points_by_density(gmm, data, percentile=5):
    """Classify points based on density threshold."""
    log_probs = gmm.score_samples(data)
    probs = np.exp(log_probs)
    threshold = np.percentile(probs, percentile)
    inside_mask = probs > threshold
    return inside_mask, ~inside_mask


def process_and_reproject_points(gmm, data, pca_mean, pca_components, tangent_mean, percentile=5):
    inside_mask, outside_mask = classify_points_by_density(gmm, data, percentile)
    
    inside_points = data[inside_mask]
    outside_points = data[outside_mask]
    
    print("Number of points inside:", len(inside_points))
    print("Number of points outside:", len(outside_points))
    
    original_inside_points = reproject_points(inside_points, pca_mean, pca_components, tangent_mean)
    
    print("Reprojected Inside Points:", original_inside_points)
    
    return inside_points, outside_points, original_inside_points


def test():
    # Sample 50 points from inside and outside
    num_samples = 50
    sampled_inside_points = inside_points[np.random.choice(len(inside_points), num_samples, replace=False)]
    sampled_outside_points = outside_points[np.random.choice(len(outside_points), num_samples, replace=False)]

    print("Number of points inside:", len(inside_points))
    print("Number of points outside:", len(outside_points))
    print("Sampled Inside Points:", sampled_inside_points)
    print("Sampled Outside Points:", sampled_outside_points)

    # Reproject sampled inside points
    original_sampled_inside_points = reproject_points(sampled_inside_points, pca_mean, pca_components, tangent_mean)

    print("Reprojected Sampled Inside Points:", original_sampled_inside_points)



if __name__ == "__main__":
    #testing 
    reduced_data = np.random.rand(100, 2)
    gmm = run_gmm_model(reduced_data)
    inside_points, outside_points, original_inside_points = process_and_reproject_points(gmm, reduced_data, pca_mean, pca_components, tangent_mean)
    test()

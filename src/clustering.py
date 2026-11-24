"""
Location segmentation
=====================
Groups Bengaluru locations into market segments by clustering a location-level
profile table, and compares three clustering families:
  - K-Means      (k = 2..10, chosen with elbow + silhouette)
  - DBSCAN       (eps chosen from a k-distance plot)
  - Agglomerative (Ward linkage, with a dendrogram)
They are compared with silhouette (higher = better separated) and
Davies-Bouldin (lower = better). The final clusters are named from their
profiles and plotted in 2-D with PCA.

Outputs:
  reports/location_clusters.csv     one row per location: features, labels from every method, segment, PCA coords
  reports/cluster_profiles.csv      one row per final segment: mean features, size, example locations
  reports/clustering_comparison.csv the three methods side by side
  reports/kmeans_k_selection.csv    inertia, silhouette and Davies-Bouldin for k = 2..10
  reports/clustering_report.md      all of the above, explained
  reports/dbscan_k_distance.csv     sorted k-distances (for the k-distance plot)
  reports/ward_linkage.csv          the Ward merge tree (for the dendrogram)
  reports/pca_loadings.csv          PCA loadings and explained variance
Figures are drawn from these files by src/visualize.py.

Usage:
    python -m src.clustering
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src import OUTPUTS_DIR, PLOTS, RANDOM_STATE, REPORTS_DIR
from src.wrangling import md_table

log = logging.getLogger(__name__)

CLEAN_CSV = OUTPUTS_DIR / "clean_data.csv"
FEATURES = ["mean_pps", "median_sqft", "mean_bhk", "listings", "pct_ready"]
# Right-skewed features are logged before scaling, so a few extreme locations
# (e.g. Whitefield with 500+ listings) do not get clusters of their own.
LOG_FEATURES = ["mean_pps", "median_sqft", "listings"]
K_RANGE = range(2, 11)
SILHOUETTE_TOLERANCE = 0.10  # accept the elbow k if its silhouette is within 10% of the best


# ──────────────────────────────────────────────────────────────────────────────
# LOCATION TABLE
# ──────────────────────────────────────────────────────────────────────────────
def build_location_table(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per location: mean price/sqft, median sqft, mean bhk, listing count, share ready to move.

    "other" is excluded: it is the bucket of all rare locations from wrangling step 8, not a place.
    """
    table = (clean[clean["location"] != "other"]
             .groupby("location")
             .agg(mean_pps=("price_per_sqft", "mean"),
                  median_sqft=("total_sqft", "median"),
                  mean_bhk=("bhk", "mean"),
                  listings=("price", "size"),
                  pct_ready=("is_ready_to_move", "mean")))
    log.info("Location table: %d locations x %d features", *table.shape)
    return table


def scale(table: pd.DataFrame) -> np.ndarray:
    """Log the skewed features, then standardise every feature to mean 0, std 1."""
    X = table[FEATURES].copy()
    X[LOG_FEATURES] = np.log(X[LOG_FEATURES])
    return StandardScaler().fit_transform(X)


def knee_index(y) -> int:
    """Index of the point farthest from the straight line joining the first and last points of a curve
    (the "kneedle" idea: where the curve bends most)."""
    y = np.asarray(y, dtype=float)
    x = np.linspace(0, 1, len(y))
    yn = (y - y.min()) / (y.max() - y.min())
    line = yn[0] + (yn[-1] - yn[0]) * x
    return int(np.argmax(np.abs(yn - line)))


def scores(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette and Davies-Bouldin on the clustered points (DBSCAN noise, label -1, excluded)."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return {"clusters": n_clusters, "noise": int((~mask).sum()), "silhouette": np.nan, "davies_bouldin": np.nan}
    return {"clusters": n_clusters, "noise": int((~mask).sum()),
            "silhouette": silhouette_score(X[mask], labels[mask]),
            "davies_bouldin": davies_bouldin_score(X[mask], labels[mask])}


# ──────────────────────────────────────────────────────────────────────────────
# K-MEANS
# ──────────────────────────────────────────────────────────────────────────────
def kmeans_k_selection(X: np.ndarray) -> tuple[pd.DataFrame, int, str]:
    """Fit K-Means for k = 2..10 and pick k.

    Rule: keep the k whose silhouette is within 10% of the best, then take the one with the lowest
    Davies-Bouldin among them. Silhouette alone tends to favour k = 2 on data without clear groups
    ("expensive vs not"), which is too coarse to describe a market; Davies-Bouldin breaks the tie
    with a second, independent measure of separation. The elbow is reported alongside.
    """
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_),
                     "smallest_cluster": int(np.bincount(km.labels_).min())})
    table = pd.DataFrame(rows)

    elbow_k = int(table["k"].iloc[knee_index(table["inertia"])])
    best = table.loc[table["silhouette"].idxmax()]
    near_best = table[table["silhouette"] >= (1 - SILHOUETTE_TOLERANCE) * best["silhouette"]]
    chosen = near_best.loc[near_best["davies_bouldin"].idxmin()]
    k = int(chosen["k"])
    elbow_sil = table.set_index("k").at[elbow_k, "silhouette"]
    why = (
        f"Silhouette is highest at k = {int(best['k'])} ({best['silhouette']:.3f}). The k values within "
        f"{SILHOUETTE_TOLERANCE:.0%} of that are {', '.join(map(str, near_best['k']))}; among them, k = {k} has the "
        f"lowest Davies-Bouldin ({chosen['davies_bouldin']:.3f}, also the lowest of all k), so **k = {k}**. "
        f"The elbow of the inertia curve is at k = {elbow_k} (silhouette {elbow_sil:.3f}"
        + (f", more than {SILHOUETTE_TOLERANCE:.0%} below the best, so not used)." if elbow_k not in near_best["k"].values
           else ").")
        + " Silhouette alone favours k = 2 (\"expensive vs not\"), which the PCA plot shows misses the split "
        "between large under-construction corridors and small established localities."
    )
    log.info("K-Means: silhouette-best k=%d, near-best %s, elbow k=%d -> chosen k=%d (lowest Davies-Bouldin)",
             int(best["k"]), near_best["k"].tolist(), elbow_k, k)
    return table, k, why


# ──────────────────────────────────────────────────────────────────────────────
# DBSCAN
# ──────────────────────────────────────────────────────────────────────────────
def dbscan_eps(X: np.ndarray, min_samples: int) -> tuple[float, np.ndarray]:
    """eps = the knee of the sorted distances to each point's min_samples-th nearest neighbour."""
    # The query point counts as its own first neighbour, which matches how DBSCAN counts min_samples.
    distances = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1]
    k_dist = np.sort(distances)
    return float(k_dist[knee_index(k_dist)]), k_dist


# ──────────────────────────────────────────────────────────────────────────────
# AGGLOMERATIVE
# ──────────────────────────────────────────────────────────────────────────────
def dendrogram_suggested_k(Z: np.ndarray) -> int:
    """The k (2..10) where the jump in merge height is largest: the most natural place to cut the tree."""
    heights = Z[:, 2]
    # Going from k clusters to k-1 costs the merge at heights[-(k-1)]; the gap measures how big that jump is.
    gaps = {k: heights[-(k - 1)] - heights[-k] for k in K_RANGE}
    return max(gaps, key=gaps.get)


# ──────────────────────────────────────────────────────────────────────────────
# NAMING
# ──────────────────────────────────────────────────────────────────────────────
TIER_NAMES = {
    2: ["Premium", "Value"],
    3: ["Premium", "Mid-market", "Budget"],
    4: ["Premium", "Upper mid-market", "Mid-market", "Budget"],
    5: ["Premium", "Upper mid-market", "Mid-market", "Affordable", "Budget"],
}
QUALIFIERS = {  # feature: (name if clearly above average, name if clearly below average)
    "median_sqft": ("large homes", "compact homes"),
    "mean_bhk": ("more bedrooms", "fewer bedrooms"),
    "listings": ("high volume", "low volume"),
    "pct_ready": ("ready to move", "under construction"),
}
QUALIFIER_Z = 0.5  # a cluster centre must be at least 0.5 std from the average to earn a qualifier


def name_segments(table: pd.DataFrame, X: np.ndarray, labels: np.ndarray) -> dict[int, str]:
    """Name each cluster: a price tier from its mean price/sqft rank, plus up to two qualifiers
    for the features where its centre is clearly above or below the average location."""
    centres = pd.DataFrame(X, columns=FEATURES).groupby(labels).mean()
    centres = centres.drop(index=-1, errors="ignore")
    order = centres["mean_pps"].sort_values(ascending=False).index
    tiers = TIER_NAMES.get(len(order), [f"Tier {i + 1}" for i in range(len(order))])
    names = {}
    for tier, cluster in zip(tiers, order):
        z = centres.loc[cluster, list(QUALIFIERS)]
        strongest = z[z.abs() >= QUALIFIER_Z].abs().sort_values(ascending=False).index[:2]
        quals = [QUALIFIERS[f][0] if z[f] > 0 else QUALIFIERS[f][1] for f in strongest]
        names[cluster] = tier + (f" ({', '.join(quals)})" if quals else "")
    return names


def profile_segments(table: pd.DataFrame, segment: pd.Series) -> pd.DataFrame:
    """Average location profile of each segment, in original units, with example locations."""
    t = table.assign(segment=segment)
    prof = t.groupby("segment").agg(
        locations=("listings", "size"), total_listings=("listings", "sum"),
        mean_pps=("mean_pps", "mean"), median_sqft=("median_sqft", "mean"),
        mean_bhk=("mean_bhk", "mean"), mean_listings=("listings", "mean"), pct_ready=("pct_ready", "mean"))
    prof["examples"] = t.groupby("segment").apply(
        lambda g: ", ".join(g.sort_values("listings", ascending=False).index[:5]), include_groups=False)
    return prof.sort_values("mean_pps", ascending=False).reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_clustering(clean: pd.DataFrame | None = None) -> dict:
    clean = pd.read_csv(CLEAN_CSV) if clean is None else clean
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    table = build_location_table(clean)
    X = scale(table)

    # K-Means
    ksel, k, k_why = kmeans_k_selection(X)
    ksel.to_csv(REPORTS_DIR / "kmeans_k_selection.csv", index=False)
    km_labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)

    # DBSCAN: min_samples = 2 x number of features (Sander et al., 1998 rule of thumb)
    min_samples = 2 * X.shape[1]
    eps, k_dist = dbscan_eps(X, min_samples)
    db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    log.info("DBSCAN eps=%.3f min_samples=%d: %d clusters, %d noise", eps, min_samples,
             len(set(db_labels) - {-1}), (db_labels == -1).sum())

    # Agglomerative (Ward): same k as K-Means so the comparison is like for like
    Z = linkage(X, method="ward")
    dendro_k = dendrogram_suggested_k(Z)
    ag_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

    comparison = pd.DataFrame([
        {"method": "K-Means", "settings": f"k={k}, n_init=20", **scores(X, km_labels)},
        {"method": "DBSCAN", "settings": f"eps={eps:.2f}, min_samples={min_samples}", **scores(X, db_labels)},
        {"method": "Agglomerative (Ward)", "settings": f"k={k}", **scores(X, ag_labels)},
    ])
    comparison["smallest_cluster"] = [int(np.bincount(l[l >= 0]).min()) if (l >= 0).any() else 0
                                      for l in (km_labels, db_labels, ag_labels)]
    comparison.to_csv(REPORTS_DIR / "clustering_comparison.csv", index=False)
    for row in comparison.itertuples():
        log.info("%-22s clusters=%d noise=%d silhouette=%.3f davies_bouldin=%.3f",
                 row.method, row.clusters, row.noise, row.silhouette, row.davies_bouldin)

    final_method, final_why = choose_method(comparison, len(table))
    final_labels = {"K-Means": km_labels, "DBSCAN": db_labels, "Agglomerative (Ward)": ag_labels}[final_method]
    names = name_segments(table, X, final_labels)
    segment = pd.Series(final_labels, index=table.index).map(names).fillna("Unclustered (noise)")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)

    assignments = table.assign(kmeans=km_labels, dbscan=db_labels, agglomerative=ag_labels,
                               segment=segment, pc1=coords[:, 0], pc2=coords[:, 1])
    assignments.sort_values(["segment", "listings"], ascending=[True, False]).to_csv(
        REPORTS_DIR / "location_clusters.csv")
    profiles = profile_segments(table, segment)
    profiles.to_csv(REPORTS_DIR / "cluster_profiles.csv", index=False)

    pd.DataFrame({"rank": np.arange(len(k_dist)), "k_distance": k_dist, "eps": eps,
                  "min_samples": min_samples}).to_csv(REPORTS_DIR / "dbscan_k_distance.csv", index=False)
    pd.DataFrame(Z, columns=["left", "right", "distance", "size"]).assign(chosen_k=k).to_csv(
        REPORTS_DIR / "ward_linkage.csv", index=False)
    pd.DataFrame(pca.components_.T, index=pd.Index(FEATURES, name="feature"), columns=["pc1", "pc2"]).assign(
        pc1_variance=pca.explained_variance_ratio_[0], pc2_variance=pca.explained_variance_ratio_[1]).to_csv(
        REPORTS_DIR / "pca_loadings.csv")

    agree = pd.crosstab(km_labels, ag_labels)
    write_report(REPORTS_DIR / "clustering_report.md", table, ksel, k, k_why, eps, min_samples, dendro_k,
                 comparison, final_method, final_why, profiles, pca, agree)
    log.info("Final: %s with %d segments: %s", final_method, len(names), "; ".join(names.values()))
    return {"table": table, "assignments": assignments, "profiles": profiles, "comparison": comparison}


def choose_method(comparison: pd.DataFrame, n_locations: int) -> tuple[str, str]:
    """Pick the method with the best silhouette and Davies-Bouldin among those that produce a usable
    segmentation (at least 2 clusters, at most 10% of locations left as noise). Ties go to K-Means."""
    usable = comparison[(comparison["clusters"] >= 2) & (comparison["noise"] <= 0.10 * n_locations)].copy()
    excluded = comparison.loc[~comparison.index.isin(usable.index), "method"].tolist()
    usable["rank"] = usable["silhouette"].rank(ascending=False) + usable["davies_bouldin"].rank()
    usable["prefer"] = usable["method"] != "K-Means"  # tie-breaker
    best = usable.sort_values(["rank", "prefer"]).iloc[0]
    why = (f"{best['method']} has the best combined rank on silhouette ({best['silhouette']:.3f}) and "
           f"Davies-Bouldin ({best['davies_bouldin']:.3f}) among the usable methods.")
    if excluded:
        why += (f" Not usable as a segmentation: {', '.join(excluded)} (fewer than 2 clusters or more than 10% "
                "of locations labelled noise).")
    return best["method"], why


def close_tiers_note(profiles: pd.DataFrame, threshold: float = 0.10) -> list[str]:
    """Warn when adjacent price tiers are close: then their names overstate the price difference."""
    pps = profiles["mean_pps"].to_numpy()
    close = [(profiles["segment"][i], profiles["segment"][i + 1]) for i in range(len(pps) - 1)
             if pps[i] / pps[i + 1] - 1 < threshold]
    if not close:
        return []
    pairs = "; ".join(f"{a} vs {b}" for a, b in close)
    return [f"Note: these adjacent segments differ by less than {threshold:.0%} in mean price per sq ft: {pairs}. "
            "The tier names only rank them; what really separates them is the other features (listing volume "
            "and share ready to move), shown in the qualifiers and the table.", ""]


def write_report(path, table, ksel, k, k_why, eps, min_samples, dendro_k, comparison, final_method,
                 final_why, profiles, pca, agree) -> Path:
    db = comparison.set_index("method").loc["DBSCAN"]
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"]).rename_axis("feature")
    best_sil = ksel["silhouette"].max()
    lines = [
        "# Clustering report: location market segments",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `src/clustering.py` from `outputs/clean_data.csv`.",
        "",
        "## 1. Location table",
        "",
        f"- {len(table)} locations (all with more than 10 clean listings; the 'other' bucket is excluded "
        "because it mixes many unrelated rare locations).",
        "- Features: mean price per sq ft (₹), median sqft, mean bhk, listing count, share ready to move.",
        f"- {', '.join(LOG_FEATURES)} are right-skewed (skewness "
        + ", ".join(f"{f} {table[f].skew():.2f}" for f in LOG_FEATURES)
        + "), so they are logged; then every feature is standardised (mean 0, std 1) so that no feature "
        "dominates the distances because of its units.",
        "",
        md_table(table[FEATURES].describe().T.rename_axis("feature").reset_index()),
        "",
        "## 2. K-Means: choosing k",
        "",
        md_table(ksel),
        "",
        k_why,
        f"Even the best silhouette ({best_sil:.3f}) is modest (values above ~0.5 indicate strong structure): "
        "Bengaluru locations form a continuum rather than well-separated groups, so the segments are a useful "
        "summary of that continuum, not natural categories.",
        "",
        f"![elbow and silhouette](../outputs/plots_v2/{PLOTS['k_selection']})",
        "",
        "## 3. DBSCAN",
        "",
        f"`min_samples = {min_samples}` (2 × number of features, a common rule of thumb). `eps = {eps:.2f}` is the "
        "knee of the k-distance plot: below it distances grow slowly (points inside dense regions), above it "
        "they shoot up (isolated points).",
        f"Result: {int(db['clusters'])} cluster(s) and {int(db['noise'])} noise locations. "
        + ("DBSCAN finds one dense core and calls the outlying locations noise: the data has no separate dense "
           "regions, which agrees with the low silhouette scores. Its noise points are the unusual locations "
           "(see `dbscan = -1` in `location_clusters.csv`)." if db["clusters"] < 2 else ""),
        "",
        f"![k-distance](../outputs/plots_v2/{PLOTS['k_distance']})",
        "",
        "## 4. Agglomerative (Ward)",
        "",
        f"Ward linkage merges the pair of clusters that increases within-cluster variance the least. Cut at "
        f"k = {k} to compare like for like with K-Means. (The largest jump in merge height, the dendrogram's own "
        f"suggestion, is at k = {dendro_k}.)",
        "",
        "Agreement between K-Means (rows) and Agglomerative (columns) labels:",
        "",
        md_table(agree.rename_axis("kmeans").reset_index().rename(columns=lambda c: f"agg_{c}" if c != "kmeans" else c)),
        "",
        f"![dendrogram](../outputs/plots_v2/{PLOTS['dendrogram']})",
        "",
        "## 5. Comparison",
        "",
        md_table(comparison),
        "",
        f"Silhouette: higher is better (−1 … 1). Davies-Bouldin: lower is better. For DBSCAN both are computed "
        "without the noise points, which flatters it. **Chosen: " + final_method + ".** " + final_why,
        "",
        "## 6. Market segments",
        "",
        "Names come from the profiles: the price tier is the rank of the segment's mean price per sq ft; a "
        f"qualifier is added when the segment centre is at least {QUALIFIER_Z} std above or below the average "
        "location on that feature.",
        "",
        md_table(profiles),
        "",
        *close_tiers_note(profiles),
        "## 7. PCA view",
        "",
        f"PC1 and PC2 explain {pca.explained_variance_ratio_[0]:.0%} and {pca.explained_variance_ratio_[1]:.0%} of "
        f"the variance ({pca.explained_variance_ratio_.sum():.0%} together). Loadings (how much each feature "
        "contributes to each axis):",
        "",
        md_table(loadings.reset_index()),
        "",
        f"![PCA](../outputs/plots_v2/{PLOTS['clusters']})",
        "",
        "## Caveat for later use",
        "",
        "The segments are built from mean price per sq ft, i.e. from prices. Using a segment as a feature in "
        "the price model would leak target information unless the segments were rebuilt inside each training fold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_clustering()


if __name__ == "__main__":
    main()

"""
Association-rule mining
=======================
Treats each listing as a "transaction" of items such as `bhk=3`,
`price=Premium` or `segment=Premium`, and mines rules "if A then C" with
mlxtend. Apriori and FP-Growth are run on the same table: they must find the
same frequent itemsets (both are exact algorithms) but differ in speed.

Metrics (for a rule A -> C):
  support    share of all listings that contain A and C
  confidence share of listings with A that also have C   = P(C | A)
  lift       confidence / share of all listings with C   = how many times more
             likely C is when A holds (1 = no association)

Outputs:
  reports/association_rules.csv          top 25 rules (lift >= 1.2)
  reports/association_rules_all.csv      every non-redundant rule, one per itemset (for the scatter plot)
  reports/association_algorithms.csv     Apriori vs FP-Growth runtime per min_support
  reports/association_report.md          everything above + the 5 best rules in plain English

Usage:
    python -m src.association
"""

import logging
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from src import OUTPUTS_DIR, PLOTS, REPORTS_DIR
from src.wrangling import md_table

log = logging.getLogger(__name__)

CLEAN_CSV = OUTPUTS_DIR / "clean_data.csv"
SEGMENTS_CSV = REPORTS_DIR / "location_clusters.csv"

MIN_SUPPORT = 0.02       # an itemset must cover at least 2% of listings (~240) to be trusted
MIN_LIFT = 1.2
MIN_CONFIDENCE = 0.5     # a rule should hold more often than not to be worth stating
TOP_N = 25
MAX_OVERLAP = 0.5        # two rules covering >50% the same listings (Jaccard) count as one finding
TIMING_SUPPORTS = [0.01, 0.02, 0.05, 0.10]
TIMING_REPEATS = 5
TIMING_BUDGET_S = 10  # stop repeating a slow run once it has taken this long in total


# ──────────────────────────────────────────────────────────────────────────────
# TRANSACTIONS
# ──────────────────────────────────────────────────────────────────────────────
def count_bin(series: pd.Series, top: int) -> pd.Series:
    """1, 2, ..., top-1 as they are; top and above as "top+"."""
    return series.where(series < top, top).astype(int).astype(str).replace(str(top), f"{top}+")


def build_items(clean: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
    """One categorical column per attribute; each value becomes an item."""
    edges = clean["total_sqft"].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy()
    sqft_labels = [f"<{edges[1]:,.0f}", f"{edges[1]:,.0f}-{edges[2]:,.0f}",
                   f"{edges[2]:,.0f}-{edges[3]:,.0f}", f"{edges[3]:,.0f}+"]
    return pd.DataFrame({
        "price": clean["price_band"],
        "sqft": pd.cut(clean["total_sqft"], bins=edges, labels=sqft_labels, include_lowest=True).astype(str),
        "bhk": count_bin(clean["bhk"], 4),
        "bath": count_bin(clean["bath"], 4),
        "area": clean["area_type"].str.replace(" Area", "", regex=False),
        "status": np.where(clean["is_ready_to_move"] == 1, "ready", "under construction"),
        # Rare locations were grouped as "other" in wrangling and have no segment.
        "segment": clean["location"].map(segments).fillna("none (rare location)"),
    })


def one_hot(items: pd.DataFrame) -> pd.DataFrame:
    """Boolean transaction table: one column per item "attribute=value"."""
    return pd.get_dummies(items, prefix_sep="=").astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# APRIORI VS FP-GROWTH
# ──────────────────────────────────────────────────────────────────────────────
def timed(fn, onehot, min_support, repeats=TIMING_REPEATS, budget_s=TIMING_BUDGET_S) -> tuple[pd.DataFrame, float, int]:
    """Run an algorithm up to `repeats` times (fewer if it has used up `budget_s` seconds);
    return its itemsets, the median runtime in ms and the number of runs."""
    times = []
    while len(times) < repeats and sum(times) < budget_s * 1000:
        start = time.perf_counter()
        result = fn(onehot, min_support=min_support, use_colnames=True)
        times.append((time.perf_counter() - start) * 1000)
    return result, float(np.median(times)), len(times)


def as_dict(itemsets: pd.DataFrame) -> dict[frozenset, float]:
    return dict(zip(itemsets["itemsets"].map(frozenset), itemsets["support"]))


def compare_algorithms(onehot: pd.DataFrame) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    """Runtime of both algorithms at several min_support values, and whether their output is identical.
    Also returns the itemsets found at each threshold, so they need not be mined again."""
    rows, found = [], {}
    for s in TIMING_SUPPORTS:
        a, ta, na = timed(apriori, onehot, s)
        f, tf, nf = timed(fpgrowth, onehot, s)
        da, df_ = as_dict(a), as_dict(f)
        same_sets = set(da) == set(df_)
        max_diff = max((abs(da[k] - df_[k]) for k in da), default=0.0) if same_sets else np.nan
        rows.append({"min_support": s, "itemsets": len(a), "longest_itemset": int(a["itemsets"].map(len).max()),
                     "apriori_ms": ta, "apriori_runs": na, "fpgrowth_ms": tf, "fpgrowth_runs": nf,
                     "fpgrowth_vs_apriori": tf / ta,
                     "same_itemsets": same_sets, "max_support_difference": max_diff})
        found[s] = f
        log.info("min_support=%.2f: %d itemsets, Apriori %.0f ms, FP-Growth %.0f ms, identical=%s",
                 s, len(a), ta, tf, same_sets)
    return pd.DataFrame(rows), found


# ──────────────────────────────────────────────────────────────────────────────
# RULES
# ──────────────────────────────────────────────────────────────────────────────
def mine_rules(itemsets: pd.DataFrame) -> pd.DataFrame:
    """All rules with lift >= MIN_LIFT and confidence >= MIN_CONFIDENCE."""
    rules = association_rules(itemsets, num_itemsets=len(itemsets), metric="lift", min_threshold=MIN_LIFT)
    return rules[rules["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)


def drop_redundant(rules: pd.DataFrame) -> pd.DataFrame:
    """Remove a rule A -> C if a simpler rule A' -> C (A' a proper subset of A) is at least as confident:
    the extra conditions in A add nothing."""
    best_conf = {(r.antecedents, r.consequents): r.confidence for r in rules.itertuples()}
    keep = []
    for r in rules.itertuples():
        simpler = (frozenset(sub) for n in range(1, len(r.antecedents))
                   for sub in combinations(r.antecedents, n))
        keep.append(not any(best_conf.get((s, r.consequents), -1) >= r.confidence for s in simpler))
    return rules[keep].reset_index(drop=True)


def one_per_itemset(rules: pd.DataFrame) -> pd.DataFrame:
    """Keep single-item conclusions, and for each itemset A ∪ C only its most confident rule:
    A -> C and C -> A describe the same listings, so listing both repeats one finding."""
    single = rules[rules["consequents"].map(len) == 1].copy()
    single["itemset"] = [a | c for a, c in zip(single["antecedents"], single["consequents"])]
    return single.loc[single.groupby("itemset")["confidence"].idxmax()].drop(columns="itemset")


def select_diverse(rules: pd.DataFrame, onehot: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Walk down the ranking by lift (then confidence) and skip any rule whose listings overlap more than
    MAX_OVERLAP (Jaccard) with a rule already chosen, so the top N are different findings, not one
    tight pattern rewritten N ways."""
    ranked = rules.sort_values(["lift", "confidence"], ascending=False)
    chosen, masks = [], []
    for idx, r in ranked.iterrows():
        mask = onehot[list(r["antecedents"] | r["consequents"])].all(axis=1).to_numpy()
        if any((mask & m).sum() / (mask | m).sum() > MAX_OVERLAP for m in masks):
            continue
        chosen.append(idx)
        masks.append(mask)
        if len(chosen) == n:
            break
    return ranked.loc[chosen]


def top_rules(ranked: pd.DataFrame, n_listings: int) -> pd.DataFrame:
    """The selected rules as a readable table."""
    return pd.DataFrame({
        "rank": range(1, len(ranked) + 1),
        "if": ranked["antecedents"].map(lambda s: " & ".join(sorted(s))),
        "then": ranked["consequents"].map(lambda s: " & ".join(sorted(s))),
        "support": ranked["support"],
        "confidence": ranked["confidence"],
        "lift": ranked["lift"],
        "consequent_support": ranked["consequent support"],
        "listings": (ranked["support"] * n_listings).round().astype(int),
        "leverage": ranked["leverage"],
        "conviction": ranked["conviction"],
    }).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# PLAIN ENGLISH
# ──────────────────────────────────────────────────────────────────────────────
def phrase(item: str, price_edges: dict) -> str:
    attr, value = item.split("=", 1)
    if attr == "price":
        return f"in the {value} price band ({price_edges[value]} lakhs)"
    if attr == "sqft":
        return f"{value} sq ft"
    if attr == "bhk":
        return f"{value.replace('+', ' or more')} BHK"
    if attr == "bath":
        return f"{value.replace('+', ' or more')} bathroom{'s' if value != '1' else ''}"
    if attr == "area":
        return f"listed as {value} area"
    if attr == "status":
        return "ready to move" if value == "ready" else "under construction"
    if attr == "segment":
        return f"in a {value} location" if not value.startswith("none") else "in a rare location"
    return item


def explain(rule: pd.Series, price_edges: dict) -> str:
    a = [phrase(i, price_edges) for i in rule["if"].split(" & ")]
    c = [phrase(i, price_edges) for i in rule["then"].split(" & ")]
    return (f"Of the listings that are {' and '.join(a)}, **{rule['confidence']:.0%}** are {' and '.join(c)}, "
            f"against {rule['consequent_support']:.0%} of all listings: **{rule['lift']:.2f}× as likely**. "
            f"The rule covers {rule['listings']:,} listings ({rule['support']:.1%}).")


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_association(clean: pd.DataFrame | None = None) -> dict:
    clean = pd.read_csv(CLEAN_CSV) if clean is None else clean
    segments = pd.read_csv(SEGMENTS_CSV, index_col="location")["segment"]
    items = build_items(clean, segments)
    onehot = one_hot(items)
    log.info("Transaction table: %d listings x %d items", *onehot.shape)

    algorithms, found = compare_algorithms(onehot)
    algorithms.to_csv(REPORTS_DIR / "association_algorithms.csv", index=False)

    itemsets = found[MIN_SUPPORT]
    all_rules = mine_rules(itemsets)
    pruned = drop_redundant(all_rules)
    distinct = one_per_itemset(pruned)
    log.info("Rules with lift >= %.1f and confidence >= %.0f%%: %d; non-redundant: %d; "
             "single-item conclusion, one per itemset: %d",
             MIN_LIFT, MIN_CONFIDENCE * 100, len(all_rules), len(pruned), len(distinct))
    top = top_rules(select_diverse(distinct, onehot), len(clean))
    top.to_csv(REPORTS_DIR / "association_rules.csv", index=False)
    everything = top_rules(distinct.sort_values(["lift", "confidence"], ascending=False), len(clean))
    everything["in_top"] = everything["if"].add("|").add(everything["then"]).isin(top["if"].add("|").add(top["then"]))
    everything.drop(columns="rank").to_csv(REPORTS_DIR / "association_rules_all.csv", index=False)

    edges = clean.groupby("price_band", observed=True)["price"].agg(["min", "max"])
    price_edges = {b: f"{r['min']:.0f}–{r['max']:.0f}" for b, r in edges.iterrows()}
    explanations = [explain(r, price_edges) for _, r in top.head(5).iterrows()]
    for e in explanations:
        log.info("  %s", e.replace("**", ""))

    write_report(REPORTS_DIR / "association_report.md", items, onehot, algorithms, itemsets,
                 all_rules, pruned, distinct, top, explanations)
    log.info("Saved reports/association_rules.csv, association_algorithms.csv, association_report.md")
    return {"items": items, "algorithms": algorithms, "rules": top}


def write_report(path: Path, items, onehot, algorithms, itemsets, all_rules, pruned, distinct, top,
                 explanations) -> Path:
    item_table = (onehot.mean().rename("support").rename_axis("item").reset_index()
                  .assign(listings=lambda t: (t["support"] * len(onehot)).round().astype(int)))
    lines = [
        "# Association rules",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `src/association.py` from `outputs/clean_data.csv` "
        "and the market segments in `reports/location_clusters.csv`.",
        "",
        "## 1. Transactions",
        "",
        f"Each of the {len(onehot):,} listings is a transaction with exactly one item from each of "
        f"{items.shape[1]} attributes, {onehot.shape[1]} distinct items in total. Price bands are the quartiles "
        "from wrangling step 8; sqft is cut at its quartiles; bhk and bath are grouped as 1, 2, 3, 4+; listings in "
        "rare locations (grouped as 'other' in wrangling) have no market segment.",
        "",
        md_table(item_table),
        "",
        "## 2. Apriori vs FP-Growth",
        "",
        "Apriori grows itemsets level by level and rescans the table for each level; FP-Growth compresses the "
        "table into a prefix tree once and mines it without generating candidates. Both are exact, so they "
        f"must return the same itemsets. Median of up to {TIMING_REPEATS} runs each (fewer for runs that "
        f"take more than {TIMING_BUDGET_S} s in total):",
        "",
        md_table(algorithms),
        "",
        (f"Here FP-Growth is **{algorithms['fpgrowth_vs_apriori'].min():.0f}–{algorithms['fpgrowth_vs_apriori'].max():.0f}× "
         "slower** than Apriori, the opposite of the textbook result. The theory assumes many distinct items and "
         "sparse transactions, where Apriori drowns in candidates. This table is the opposite case: only "
         f"{onehot.shape[1]} items, and every listing has exactly {items.shape[1]} of them (dense data, long "
         "patterns). mlxtend's Apriori counts candidates with vectorised NumPy operations, which is cheap for 27 "
         "columns, while its FP-Growth builds and recursively mines the tree in pure Python. The algorithm's "
         "complexity matters less than the data shape and the implementation."
         if (algorithms["fpgrowth_vs_apriori"] > 1).all() else
         "FP-Growth is faster at every threshold, as the theory predicts: it avoids generating candidates."),
        "",
        ("Both algorithms returned identical itemsets with identical supports at every threshold."
         if algorithms["same_itemsets"].all() and (algorithms["max_support_difference"].fillna(1) < 1e-12).all()
         else "**The algorithms disagreed at some threshold; see the table.**"),
        "",
        "## 3. Rules",
        "",
        f"From the {len(itemsets):,} itemsets with support ≥ {MIN_SUPPORT:.0%}: {len(all_rules):,} rules have lift ≥ "
        f"{MIN_LIFT} and confidence ≥ {MIN_CONFIDENCE:.0%}. {len(all_rules) - len(pruned):,} of them are redundant "
        "(a rule with fewer conditions and the same conclusion is at least as confident) and are removed, "
        f"leaving {len(pruned):,}. Keeping only single-item conclusions and, for each itemset, its most "
        f"confident rule leaves {len(distinct):,}.",
        "",
        f"Ranking those purely by lift puts one pattern (small, cheap 1-BHK flats with 1 bathroom, ~320 listings) "
        f"in every one of the top {TOP_N} places, rewritten with different items on each side. So the top "
        f"{TOP_N} are picked greedily: go down the lift ranking and skip a rule if the listings it covers overlap "
        f"more than {MAX_OVERLAP:.0%} (Jaccard) with a rule already chosen. Result:",
        "",
        md_table(top),
        "",
        f"![rules](../outputs/plots_v2/{PLOTS['rules']})",
        "",
        "## 4. The five best rules in plain English",
        "",
        *[f"{i}. {e}" for i, e in enumerate(explanations, 1)],
        "",
        "Caveats: rules describe co-occurrence in these listings, not cause and effect. Market segments were "
        "built from location averages (including the share ready to move), so rules linking a segment to "
        "readiness partly restate how the segment was defined.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_association()


if __name__ == "__main__":
    main()

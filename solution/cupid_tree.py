"""Tree-based Cupid (Madhavan, Bernstein, Rahm, VLDB 2001) working directly on XML schema trees."""

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import datetime
from math import isfinite

import pandas as pd

try:
    from nltk.corpus import wordnet

    wordnet.synsets("test")
except Exception:
    wordnet = None


class Node:
    """A schema-tree element with a parent link and optional leaf data type."""

    def __init__(self, name, parent=None):
        self.name, self.parent = name, parent
        self.children, self.dtype = [], None

    @property
    def is_leaf(self):
        return not self.children

    @property
    def path(self):
        return (self.parent.path + "." if self.parent else "") + self.name

    def post_order(self):
        # Visit descendants first so parent similarities can use completed child scores.
        for c in self.children:
            yield from c.post_order()
        yield self

    def leaves(self):
        return (
            [self] if self.is_leaf else [l for c in self.children for l in c.leaves()]
        )

    def show(self, indent=0):
        print("  " * indent + self.name + (f" : {self.dtype}" if self.is_leaf else ""))
        for c in self.children:
            c.show(indent + 1)


TYPES = ["integer", "float", "boolean", "date", "string"]


def _infer_type(values):
    def kind(v):
        # Normalize whitespace and classify each XML text value by its basic scalar type.
        v = (v or "").strip()
        if v.lower() in ("true", "false"):
            return "boolean"
        if re.fullmatch(r"[+-]?\d+", v):
            return "integer"
        if re.fullmatch(r"[+-]?\d*\.\d+", v):
            return "float"
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return "date"
        except ValueError:
            return "string"

    kinds = {kind(v) for v in values}
    # Mixed integer/float samples widen to float; other mixed types fall back to string.
    if kinds == {"integer", "float"}:
        return "float"
    return kinds.pop() if len(kinds) == 1 else "string"


def load_schema_tree(path, type_overrides=None):
    """Load a merged XML instance tree and optionally use declared leaf types.

    Type overrides are keyed by dotted element path, e.g.
    ``PO.POHeader.PONumber``. Unspecified leaf types are inferred from samples.
    """
    if type_overrides is not None and not isinstance(type_overrides, Mapping):
        raise TypeError("Type overrides must be a mapping of element paths to types.")

    xml_root = ET.parse(path).getroot()
    records = list(xml_root)
    root = Node(records[0].tag)
    samples = {}

    def merge(xml_node, node):
        # Merge each record into one tree, reusing same-named children across records.
        for child in xml_node:
            match = next((c for c in node.children if c.name == child.tag), None)
            if match is None:
                match = Node(child.tag, node)
                node.children.append(match)
            if len(child):
                merge(child, match)
            else:
                # Keep all observed leaf values so type inference reflects the records.
                samples.setdefault(match, []).append(child.text)

    for rec in records:
        merge(rec, root)

    leaf_paths = {leaf.path for leaf in samples}
    if type_overrides is not None:
        invalid_keys = [
            path for path in type_overrides if not isinstance(path, str) or not path
        ]
        if invalid_keys:
            raise ValueError("Type override paths must be non-empty strings.")
        invalid_paths = set(type_overrides) - leaf_paths
        if invalid_paths:
            raise ValueError(
                "Type overrides must refer to leaf paths in the XML tree: "
                + ", ".join(sorted(invalid_paths))
            )
        invalid_types = {
            path: dtype
            for path, dtype in type_overrides.items()
            if not isinstance(dtype, str) or dtype not in TYPES
        }
        if invalid_types:
            raise ValueError(
                "Type overrides must use supported types "
                f"{TYPES}; invalid entries: {invalid_types}"
            )

    for leaf, vals in samples.items():
        leaf.dtype = (
            type_overrides[leaf.path]
            if type_overrides is not None and leaf.path in type_overrides
            else _infer_type(vals)
        )
    return root


# ---- linguistic matching -------------------------------------------------
def tokenize(name):
    # Split camelCase first, then split on punctuation, underscores, and spaces.
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", name) if t]


def _token_sim(a, b):
    # Exact names score highest; optional WordNet and substring matches provide weaker evidence.
    if a == b:
        return 1.0
    if wordnet is not None:
        sa = {l.name().lower() for s in wordnet.synsets(a) for l in s.lemmas()}
        if b in sa:
            return 0.8
    if len(a) > 2 and len(b) > 2 and (a in b or b in a):
        return 0.6
    return 0.0


def _build_category_index(thesaurus):
    """Map normalized token phrases to their shared thesaurus category."""
    if thesaurus is None:
        return None

    categories = {}
    for category, aliases in thesaurus.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Thesaurus category names must be non-empty strings.")
        if not isinstance(aliases, (list, tuple)) or not aliases:
            raise ValueError(f"Thesaurus category {category!r} must have aliases.")
        for alias in [category, *aliases]:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("Thesaurus aliases must be non-empty strings.")
            phrase = tuple(tokenize(alias))
            if not phrase:
                raise ValueError(f"Thesaurus alias {alias!r} has no tokens.")
            existing = categories.get(phrase)
            if existing is not None and existing != category:
                raise ValueError(
                    f"Thesaurus alias {alias!r} belongs to multiple categories."
                )
            categories[phrase] = category
    return sorted(categories.items(), key=lambda item: len(item[0]), reverse=True)


def _categorize_name(name, category_index):
    """Replace recognized token phrases with their shared category identifiers."""
    tokens = tokenize(name)
    categorized = []
    i = 0
    while i < len(tokens):
        match = next(
            (
                (phrase, category)
                for phrase, category in category_index
                if tuple(tokens[i : i + len(phrase)]) == phrase
            ),
            None,
        )
        if match is None:
            categorized.append(tokens[i])
            i += 1
        else:
            phrase, category = match
            categorized.append(category)
            i += len(phrase)
    return categorized


def name_sim(n1, n2, category_index=None):
    # Cupid treats the token collections as sets, so repeated tokens count once.
    if category_index is None:
        t1, t2 = tokenize(n1), tokenize(n2)
    else:
        t1, t2 = (
            _categorize_name(n1, category_index),
            _categorize_name(n2, category_index),
        )
    t1, t2 = sorted(set(t1)), sorted(set(t2))
    if not t1 or not t2:
        return 0.0
    # Average every token's best cross-name match, weighted by the total token count.
    best_from_t1 = sum(max(_token_sim(a, b) for b in t2) for a in t1)
    best_from_t2 = sum(max(_token_sim(b, a) for a in t1) for b in t2)
    return (best_from_t1 + best_from_t2) / (len(t1) + len(t2))


# The paper specifies [0, 0.5] compatibility and 0.5 for identical types, but
# does not publish the off-diagonal table entries. This conservative baseline
# assigns no compatibility to different types.
TYPE_COMPAT = pd.DataFrame(
    [[0.50 if row == column else 0.2 for column in TYPES] for row in TYPES],
    index=TYPES,
    columns=TYPES,
)


def type_compat(t1, t2):
    # Missing or unsupported types provide no type-based evidence for a match.
    if t1 not in TYPE_COMPAT.index or t2 not in TYPE_COMPAT.columns:
        return 0.0
    return float(TYPE_COMPAT.loc[t1, t2])


# ---- TreeMatch -----------------------------------------------------------
def cupid_tree_match(
    source,
    target,
    w_struct=0.6,
    th_accept=0.5,
    th_high=0.6,
    th_low=0.35,
    c_inc=1.2,
    c_dec=0.9,
    thesaurus=None,
    **_,
):
    """Match schema leaves using Cupid's bottom-up tree-similarity procedure.

    Linguistic matching uses token categories when a thesaurus is provided;
    without one, it remains a lightweight token-based approximation.
    """
    if not 0.0 <= w_struct <= 1.0:
        raise ValueError("Structural weights must be between 0 and 1.")
    if not 0.0 <= th_low < th_accept < th_high <= 1.0:
        raise ValueError(
            "Thresholds must satisfy 0 <= th_low < th_accept < th_high <= 1."
        )
    if not isfinite(c_inc) or not isfinite(c_dec) or c_inc < 0.0 or c_dec < 0.0:
        raise ValueError(
            "Structural reinforcement factors must be finite and non-negative."
        )

    # Post-order lists allow structural similarities to be propagated bottom-up.
    s_nodes, t_nodes = list(source.post_order()), list(target.post_order())
    category_index = _build_category_index(thesaurus)

    # Compute the simplified pairwise linguistic score for every node pair.
    lsim = {}
    for s in s_nodes:
        for t in t_nodes:
            lsim[s, t] = name_sim(s.name, t.name, category_index)

    # Leaf structural similarity starts from type compatibility; all pairs use w_struct.
    ssim, wsim = {}, {}
    for s in s_nodes:
        for t in t_nodes:
            ssim[s, t] = (
                type_compat(s.dtype, t.dtype) if (s.is_leaf and t.is_leaf) else 0.0
            )
            wsim[s, t] = w_struct * ssim[s, t] + (1 - w_struct) * lsim[s, t]

    for s in s_nodes:
        for t in t_nodes:
            if s.is_leaf and t.is_leaf:
                continue
            # Compare the fraction of each node's descendant leaves supported by a strong pair.
            ls, lt = s.leaves(), t.leaves()
            # Avoid unlikely structural matches when subtree sizes differ substantially.
            if max(len(ls), len(lt)) > 2 * min(len(ls), len(lt)):
                continue
            strong_s = sum(any(wsim[a, b] > th_accept for b in lt) for a in ls)
            strong_t = sum(any(wsim[a, b] > th_accept for a in ls) for b in lt)
            ssim[s, t] = (strong_s + strong_t) / (len(ls) + len(lt))
            wsim[s, t] = w_struct * ssim[s, t] + (1 - w_struct) * lsim[s, t]
            # Reinforce descendant leaf pairs for promising structures, or dampen weak ones.
            factor = (
                c_inc
                if wsim[s, t] > th_high
                else c_dec
                if wsim[s, t] < th_low
                else None
            )
            if factor is not None:
                for a in ls:
                    for b in lt:
                        ssim[a, b] = min(1.0, ssim[a, b] * factor)
                        wsim[a, b] = w_struct * ssim[a, b] + (1 - w_struct) * lsim[a, b]

    # Leaf mapping: every leaf pair with wsim >= th_accept, best first (n:m allowed).
    pairs = sorted(
        ((wsim[a, b], a, b) for a in source.leaves() for b in target.leaves()),
        key=lambda x: -x[0],
    )
    matches = [
        (a.path, b.path, round(score, 3)) for score, a, b in pairs if score >= th_accept
    ]
    return matches, wsim

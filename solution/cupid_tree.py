"""Tree-based Cupid (Madhavan, Bernstein, Rahm, VLDB 2001) working directly on XML schema trees."""

import re
import xml.etree.ElementTree as ET
from datetime import datetime

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


def load_schema_tree(path):
    """Root = record element; children = merged element structure of all records; leaf types inferred from values."""
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
    for leaf, vals in samples.items():
        leaf.dtype = _infer_type(vals)
    return root


# ---- linguistic matching -------------------------------------------------
def tokenize(name):
    # Split camelCase first, then split on punctuation, underscores, and spaces.
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


def name_sim(n1, n2):
    t1, t2 = tokenize(n1), tokenize(n2)
    if not t1 or not t2:
        return 0.0
    # Average best token matches in both directions so extra tokens affect the score.
    fwd = sum(max(_token_sim(a, b) for b in t2) for a in t1) / len(t1)
    bwd = sum(max(_token_sim(b, a) for a in t1) for b in t2) / len(t2)
    return (fwd + bwd) / 2


TYPES = ["integer", "float", "boolean", "date", "string"]

# Data type compatibility table in [0, 1]; symmetric. Used as the initial structural
# similarity of leaf pairs (Cupid, Section 5).
TYPE_COMPAT = pd.DataFrame(
    [
        # integer float boolean date string
        [1.0, 0.8, 0.4, 0.2, 0.3],  # integer
        [0.8, 1.0, 0.2, 0.2, 0.3],  # float
        [0.4, 0.2, 1.0, 0.1, 0.3],  # boolean
        [0.2, 0.2, 0.1, 1.0, 0.4],  # date
        [0.3, 0.3, 0.3, 0.4, 1.0],  # string
    ],
    index=TYPES,
    columns=TYPES,
)


def type_compat(t1, t2):
    return float(TYPE_COMPAT.loc[t1, t2])


# ---- TreeMatch -----------------------------------------------------------
def cupid_tree_match(
    source,
    target,
    leaf_w_struct=0.2,
    w_struct=0.2,
    th_accept=0.7,
    th_high=0.6,
    th_low=0.35,
    c_inc=1.2,
    c_dec=0.9,
    th_ns=0.7,
    **_,
):
    # Post-order lists allow structural similarities to be propagated bottom-up.
    s_nodes, t_nodes = list(source.post_order()), list(target.post_order())

    # Compute linguistic scores for every node pair, pruning very weak name matches.
    lsim = {}
    for s in s_nodes:
        for t in t_nodes:
            ns = name_sim(s.name, t.name)
            lsim[s, t] = ns if ns >= th_ns * 0.5 else 0.0

    # Leaf structural similarity starts from the data type compatibility table.
    # Combine that evidence with name similarity; inner nodes begin with name evidence only.
    ssim, wsim = {}, {}
    for s in s_nodes:
        for t in t_nodes:
            ssim[s, t] = (
                type_compat(s.dtype, t.dtype) if (s.is_leaf and t.is_leaf) else 0.0
            )
            w = leaf_w_struct if (s.is_leaf and t.is_leaf) else w_struct
            wsim[s, t] = w * ssim[s, t] + (1 - w) * lsim[s, t]

    for s in s_nodes:
        for t in t_nodes:
            if s.is_leaf and t.is_leaf:
                continue
            # Compare the fraction of each node's descendant leaves supported by a strong pair.
            ls, lt = s.leaves(), t.leaves()
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
            if factor:
                for a in ls:
                    for b in lt:
                        ssim[a, b] = min(1.0, ssim[a, b] * factor)
                        wsim[a, b] = (
                            leaf_w_struct * ssim[a, b]
                            + (1 - leaf_w_struct) * lsim[a, b]
                        )

    # Leaf mapping: every leaf pair with wsim >= th_accept, best first (n:m allowed).
    pairs = sorted(
        ((wsim[a, b], a, b) for a in source.leaves() for b in target.leaves()),
        key=lambda x: -x[0],
    )
    matches = [
        (a.path, b.path, round(score, 3)) for score, a, b in pairs if score >= th_accept
    ]
    return matches, wsim

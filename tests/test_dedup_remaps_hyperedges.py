"""Dedup must rewire hyperedge members onto survivors, not drop them.

`build()` rewires EDGE endpoints to dedup survivors, but `combined["hyperedges"]`
never went through the same remap. The member naming a merged-away id was simply
absent from the rebuilt graph, so the group lost a participant — and could fall
under the 3-member threshold that makes it a hyperedge at all — with nothing on
stderr and, crucially, **no dangling reference**, so a referential-integrity
check saw a perfectly consistent graph (#2805).

`_normalize_hyperedge_members` / `_coerce_hyperedge_member_refs` normalise member
SHAPE (bare id vs object) but never resolve a member against surviving node ids,
which is why they do not cover this.
"""
import pytest

from graphify.build import build
from graphify.dedup import _remap_hyperedge_members, deduplicate_entities


def _node(nid, label):
    return {"id": nid, "label": label, "file_type": "concept",
            "source_file": "notes/a.md"}


def _extraction(members, key="nodes"):
    """Two nodes that normalise to the same label, so dedup merges them; the
    hyperedge names the id that loses. `key` lets a test spell the member list
    with a legacy alias (`members` / `node_ids`) instead of canonical `nodes`."""
    return {
        "nodes": [
            _node("alpha_a", "Alpha Concept"),
            _node("alpha_concept_long_variant_id", "alpha concept"),
            _node("beta_node", "Beta"),
            _node("gamma_node", "Gamma"),
        ],
        "edges": [],
        "hyperedges": [{"id": "the_group", "label": "The Group",
                        key: members, "relation": "participate_in",
                        "confidence": "INFERRED", "confidence_score": 0.75,
                        "source_file": "notes/a.md"}],
    }


def _members(G):
    hes = G.graph.get("hyperedges", [])
    assert len(hes) == 1, hes
    return [m if isinstance(m, str) else m.get("id") for m in hes[0]["nodes"]]


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------

def test_member_follows_the_survivor_instead_of_vanishing():
    G = build([_extraction(
        ["alpha_concept_long_variant_id", "beta_node", "gamma_node"])])
    assert _members(G) == ["alpha_a", "beta_node", "gamma_node"]


def test_the_group_keeps_its_size():
    """The quiet part: a group of 3 became a group of 2, which can drop it below
    the threshold that makes it a hyperedge."""
    G = build([_extraction(
        ["alpha_concept_long_variant_id", "beta_node", "gamma_node"])])
    assert len(_members(G)) == 3


def test_no_member_is_left_pointing_at_a_merged_away_id():
    G = build([_extraction(
        ["alpha_concept_long_variant_id", "beta_node", "gamma_node"])])
    assert all(m in G.nodes for m in _members(G))
    assert "alpha_concept_long_variant_id" not in G.nodes


def test_object_shaped_members_are_remapped_too():
    """Members are tolerated as bare ids or as objects carrying one."""
    G = build([_extraction([
        {"id": "alpha_concept_long_variant_id", "role": "subject"},
        {"id": "beta_node"}, {"id": "gamma_node"},
    ])])
    assert _members(G) == ["alpha_a", "beta_node", "gamma_node"]


def test_an_untouched_hyperedge_is_unchanged():
    G = build([_extraction(["alpha_a", "beta_node", "gamma_node"])])
    assert _members(G) == ["alpha_a", "beta_node", "gamma_node"]


def test_an_alias_keyed_hyperedge_is_remapped_not_deleted():
    """A `members`-keyed group reaches dedup BEFORE build_from_json canonicalizes
    it (#1561 fold runs later). It must be normalized first and then rewired like
    any other hyperedge — not deleted for lacking a `nodes` list."""
    G = build([_extraction(
        ["alpha_concept_long_variant_id", "beta_node", "gamma_node"], key="members")])
    assert _members(G) == ["alpha_a", "beta_node", "gamma_node"]


# ---------------------------------------------------------------------------
# _remap_hyperedge_members directly
# ---------------------------------------------------------------------------

def test_two_members_collapsing_onto_one_survivor_dedupe():
    """They were the same entity, so one entry is right. The old code shrank the
    group AND lost the participant; this shrinks it because the members really
    were duplicates."""
    hes = [{"id": "h", "nodes": ["a_old", "a_new", "b", "c"]}]
    _remap_hyperedge_members(hes, {"a_old": "a", "a_new": "a"})
    assert hes[0]["nodes"] == ["a", "b", "c"]


def test_member_order_is_preserved():
    hes = [{"id": "h", "nodes": ["c", "b_old", "a"]}]
    _remap_hyperedge_members(hes, {"b_old": "b"})
    assert hes[0]["nodes"] == ["c", "b", "a"]


def test_object_members_keep_their_other_fields():
    """Remapping an object-shaped member must preserve its non-id fields."""
    hes = [{
        "id": "h",
        "nodes": [
            {"id": "x_old", "role": "subject"},
            {"id": "y", "role": "object"},
            {"id": "z", "role": "context"},
        ],
    }]
    _remap_hyperedge_members(hes, {"x_old": "x"})
    assert hes[0]["nodes"] == [
        {"id": "x", "role": "subject"},
        {"id": "y", "role": "object"},
        {"id": "z", "role": "context"},
    ]


@pytest.mark.parametrize("he", [
    {"id": "h"},                       # no members key
    {"id": "h", "nodes": None},        # members not a list
    {"id": "h", "nodes": []},          # empty
    {"id": "h", "nodes": [None, 7]},   # junk members
    "not-a-dict",
])
def test_malformed_hyperedges_do_not_raise(he):
    _remap_hyperedge_members([he], {"a": "b"})


def test_an_empty_remap_changes_nothing():
    hes = [{"id": "h", "nodes": ["a", "b", "c"]}]
    _remap_hyperedge_members(hes, {})
    assert hes[0]["nodes"] == ["a", "b", "c"]


def test_an_entry_without_a_nodes_list_is_passed_through_not_deleted():
    """The remap can only rewire a canonical `nodes` list. An entry it cannot
    interpret (alias-keyed here) must survive untouched for build_from_json to
    heal — the kept-list rewrite must never turn "skip" into "delete"."""
    hes = [{"id": "h", "members": ["a", "b", "c"]}]
    _remap_hyperedge_members(hes, {"a": "z"})
    assert hes == [{"id": "h", "members": ["a", "b", "c"]}]


def test_chained_collapse_lands_on_the_final_survivor():
    """A dedup remap built from union-find is fully flattened (path-compressed),
    so a member of a chained component (a_old -> a_mid -> a) rewires directly to
    the final survivor in a single lookup, never to an intermediate."""
    hes = [{"id": "h", "nodes": ["a_old", "a_mid", "b", "c"]}]
    # what components()/UnionFind produces: every non-winner maps to the winner
    _remap_hyperedge_members(hes, {"a_old": "a", "a_mid": "a"})
    assert hes[0]["nodes"] == ["a", "b", "c"]


def test_a_hyperedge_collapsing_to_one_member_is_dropped():
    """A deduplicated singleton is no longer a group relationship."""
    hes = [{"id": "h", "nodes": ["a_old", "a_new"]}]
    _remap_hyperedge_members(hes, {"a_old": "a", "a_new": "a"})
    assert hes == []


# ---------------------------------------------------------------------------
# deduplicate_entities(..., hyperedges=) must clean up on EVERY return path
# ---------------------------------------------------------------------------

_DISTINCT_NODES = [
    _node("alpha_concept_long_variant_id", "alpha concept"),
    _node("beta_node", "Beta"),
    _node("gamma_node", "Gamma"),
]


def test_a_pair_is_dropped_even_when_dedup_merges_nothing():
    """The cardinality cleanup must not depend on whether a merge happened: with
    an empty remap the early return used to skip _remap_hyperedge_members, so a
    direct caller kept a two-member "group" that every other path rejects."""
    hes = [
        {"id": "pair", "nodes": ["alpha_concept_long_variant_id", "beta_node"]},
        {"id": "trio", "nodes": ["alpha_concept_long_variant_id", "beta_node", "gamma_node"]},
    ]
    deduplicate_entities(list(_DISTINCT_NODES), [], communities={}, hyperedges=hes)
    assert [h["id"] for h in hes] == ["trio"]


def test_a_pair_is_dropped_on_the_single_node_short_circuit():
    """Same contract on the other early return (nothing to dedup with one node)."""
    hes = [{"id": "pair", "nodes": ["beta_node", "gamma_node"]}]
    deduplicate_entities([_node("beta_node", "Beta")], [], communities={}, hyperedges=hes)
    assert hes == []


def test_a_pair_is_dropped_after_duplicate_id_collapse():
    """The third early return: two records sharing one id collapse to a single
    `unique_nodes` entry, so the function short-circuits AFTER the initial
    length check and used to skip the hyperedge pass on the way out."""
    hes = [{"id": "pair", "nodes": ["only_node", "beta_node"]}]
    deduplicate_entities(
        [_node("only_node", "Only"), _node("only_node", "Only Again")],
        [], communities={}, hyperedges=hes,
    )
    assert hes == []

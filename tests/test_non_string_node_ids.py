"""Non-string node ids from LLM backends must not crash the build (#2326).

A backend can emit ``{"id": 10}`` where the schema says ``{"id": "10"}``. Every
id consumer downstream assumes ``str``, so an int id used to abort the whole
build in three different places. These tests pin the crash sites and the
edge/hyperedge linkage that a node-only coercion would silently break.
"""
import networkx as nx
import pytest

from graphify.build import build, build_from_json


def _node(nid, label, **kw):
    return {
        "id": nid,
        "label": label,
        "file_type": "concept",
        "source_file": "a.py",
        **kw,
    }


def _edge(src, tgt):
    return {"source": src, "target": tgt, "relation": "uses", "confidence": "EXTRACTED"}


def test_pick_winner_survives_int_id_in_duplicate_group():
    """dedup._pick_winner regex-searched the raw id (the issue's traceback).

    Driven through ``build`` because that is dedup's only production caller, so
    ``build`` is where the coercion has to land for this path to be fixed.
    """
    ext = {"nodes": [_node(10, "Alpha"), _node("alpha_c1", "Alpha")], "edges": []}
    G = build([ext], dedup=True)
    assert all(isinstance(nid, str) for nid in G.nodes)


def test_build_accepts_a_single_int_id_node_with_no_duplicate():
    """build_from_json's sorted(node_set) crashed even with nothing to dedup."""
    ext = {"nodes": [_node(10, "Alpha"), _node("b", "Beta")], "edges": [_edge(10, "b")]}
    G = build([ext], dedup=True)
    assert "10" in G.nodes
    assert 10 not in G.nodes


def test_int_id_endpoints_stay_connected_after_coercion():
    """Coercing node ids without coercing endpoints would orphan the edge."""
    ext = {"nodes": [_node(10, "Alpha"), _node(20, "Beta")], "edges": [_edge(10, 20)]}
    G = build([ext], dedup=True)
    assert G.has_edge("10", "20")


def test_int_id_survives_a_fuzzy_dedup_group():
    ext = {
        "nodes": [_node(10, "PaymentProcessor"), _node("b", "PaymentProcessors")],
        "edges": [_edge(10, "b")],
    }
    G = build([ext], dedup=True)
    assert all(isinstance(nid, str) for nid in G.nodes)


def test_float_id_is_coerced_too():
    ext = {"nodes": [_node(1.5, "Alpha"), _node("b", "Beta")], "edges": [_edge(1.5, "b")]}
    G = build([ext], dedup=True)
    assert G.has_edge("1.5", "b")


def test_legacy_from_to_endpoints_are_coerced():
    """dedup reads the legacy from/to aliases (#803), so they need it as well."""
    ext = {
        "nodes": [_node(10, "Alpha"), _node("b", "Beta"), _node("c", "Gamma")],
        "edges": [{"from": 10, "to": "b", "relation": "uses", "confidence": "EXTRACTED"}],
    }
    G = build([ext], dedup=True)
    assert G.has_edge("10", "b")


def test_node_id_set_coerces_numeric_ids_like_members_are():
    """#2326 heals numeric node ids to their string form, and member coercion
    does the same to member refs — so the comparison set has to be built in the
    same space. Keyed on raw values, `"7" in {7}` is False and every member of
    an otherwise valid group is dropped."""
    from graphify.build import gate_hyperedges, node_id_set

    nodes = [{"id": 7}, {"id": 8}, {"id": 9}]
    assert node_id_set(nodes) == {"7", "8", "9"}

    kept, dropped = gate_hyperedges([{"id": "g", "nodes": [7, 8, 9]}], nodes)
    assert dropped == 0, "a group over numeric node ids must survive"
    assert kept[0]["nodes"] == [7, 8, 9], (
        "the raw writers persist `nodes` unchanged, so surviving members have "
        "to come back in the node list's own id space"
    )


def test_gate_hyperedges_returns_members_in_the_node_lists_own_id_space():
    """The raw `--no-cluster` writers gate against the node records they are
    about to persist and write those records unchanged. Coercing only the
    comparison side left the file holding nodes `[7, 8, 9]` and members
    `["7", "8", "9"]` — a dangling reference, the shape #1916 removed, written
    by the gate that exists to prevent it. Compare coerced, return raw."""
    from graphify.build import gate_hyperedges
    from graphify.watch import _gated_hyperedges

    nodes = [{"id": 7}, {"id": 8}, {"id": 9}]
    written_ids = {n["id"] for n in nodes}

    kept, _ = gate_hyperedges([{"id": "g", "nodes": [7, 8, 9]}], nodes)
    assert set(kept[0]["nodes"]) <= written_ids, (
        f"members {kept[0]['nodes']} must name nodes actually written "
        f"{sorted(written_ids, key=str)}"
    )

    # watch's raw writer shares the gate and writes the same node records.
    members = _gated_hyperedges([{"id": "g", "nodes": [7, 8, 9]}], nodes)[0]["nodes"]
    assert set(members) <= written_ids


def test_prune_graph_json_sources_keeps_the_files_own_node_id_space():
    """An externally produced or legacy graph.json can carry numeric node ids.
    The pruner rewrites hyperedges but leaves the node records alone, so a
    coerced member list would turn a valid group into a dangling one on disk."""
    import json

    from graphify.cli import _prune_graph_json_sources

    graph_path = tmp_graph_json(
        nodes=[
            {"id": 7, "source_file": "a.py"},
            {"id": 8, "source_file": "a.py"},
            {"id": 9, "source_file": "a.py"},
            {"id": 99, "source_file": "gone.py"},
        ],
        hyperedges=[{"id": "g", "source_file": "a.py", "nodes": [7, 8, 9, 99]}],
    )
    _prune_graph_json_sources(graph_path, ["gone.py"])

    data = json.loads(graph_path.read_text(encoding="utf-8"))
    node_ids = {n["id"] for n in data["nodes"]}
    members = data["hyperedges"][0]["nodes"]
    assert node_ids == {7, 8, 9}, "the stale source's node is pruned"
    assert set(members) <= node_ids, (
        f"members {members} must name nodes actually written "
        f"{sorted(node_ids, key=str)}"
    )


def tmp_graph_json(*, nodes, hyperedges):
    """Write a minimal hand-authored graph.json and return its path."""
    import json
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "graph.json"
    path.write_text(
        json.dumps({"nodes": nodes, "edges": [], "hyperedges": hyperedges}),
        encoding="utf-8",
    )
    return path


def test_graph_container_membership_uses_the_coerced_id_space():
    """`attach_hyperedges`, `to_json` and `build_merge` pass the graph itself as
    the container. Coercing only the member side left `"7" in nx.Graph([7])`
    False, so every member of a valid group over numeric node ids was dropped —
    the list path was fixed by node_id_set, the container path was not."""
    import json
    import tempfile
    from pathlib import Path

    import networkx as nx

    from graphify.build import canonical_hyperedge
    from graphify.export import attach_hyperedges, to_json

    G = nx.Graph()
    G.add_nodes_from([7, 8, 9])
    assert canonical_hyperedge({"id": "g", "nodes": [7, 8, 9]}, G)["nodes"] == ["7", "8", "9"]

    H = nx.Graph()
    H.add_nodes_from([7, 8, 9])
    attach_hyperedges(H, [{"id": "g", "nodes": [7, 8, 9]}])
    assert [h["id"] for h in H.graph.get("hyperedges", [])] == ["g"]

    J = nx.Graph()
    J.add_nodes_from([7, 8, 9])
    J.graph["hyperedges"] = [{"id": "g", "nodes": [7, 8, 9]}]
    out = Path(tempfile.mkdtemp()) / "graph.json"
    to_json(J, {0: [7, 8, 9]}, str(out))
    assert [h["id"] for h in json.loads(out.read_text())["hyperedges"]] == ["g"]


def test_to_json_writes_members_in_the_graphs_own_id_space():
    """Coercing only the comparison side left graph.json internally inconsistent:
    node_link_data writes `{"id": 7}` while the surviving member reads `"7"`, so
    the written file carries a dangling member — the very shape #1916 removed.
    Whatever the gate keeps has to come back out in the node ids' own space."""
    import json
    import tempfile
    from pathlib import Path

    import networkx as nx

    from graphify.export import to_json

    G = nx.Graph()
    G.add_nodes_from([7, 8, 9])
    G.graph["hyperedges"] = [{"id": "g", "nodes": [7, 8, 9]}]
    out = Path(tempfile.mkdtemp()) / "graph.json"
    to_json(G, {0: [7, 8, 9]}, str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    node_ids = {n["id"] for n in data["nodes"]}
    assert data["hyperedges"], "the group must survive"
    members = data["hyperedges"][0]["nodes"]
    assert set(members) <= node_ids, (
        f"members {members} must name nodes actually written {sorted(node_ids, key=str)}"
    )


def test_semantic_cleanup_keeps_a_group_over_numeric_node_ids():
    """`_normalize_hyperedge_members` coerces members to strings, so the
    surviving-id set has to be built in the same space or every member of a
    valid numeric group is filtered out and the group dropped."""
    from graphify.semantic_cleanup import sanitize_semantic_fragment

    fragment = {
        "nodes": [
            {"id": n, "label": f"N{n}", "file_type": "code", "source_file": "a.py"}
            for n in (7, 8, 9)
        ],
        "edges": [],
        "hyperedges": [{"id": "g", "nodes": [7, 8, 9]}],
    }
    out = sanitize_semantic_fragment(fragment)
    assert [h["id"] for h in out["hyperedges"]] == ["g"]
    assert out["hyperedges"][0]["nodes"] == ["7", "8", "9"]


def test_prefix_graph_for_global_builds_the_relabel_map_once(monkeypatch):
    """The coerced relabel map must be built once per graph, not once per
    hyperedge — rebuilding it inside the loop makes prefixing O(nodes x
    hyperedges), and a semantic graph can carry thousands of groups."""
    import networkx as nx

    import graphify.build as buildmod

    calls = {"n": 0}
    real = buildmod._coerce_id

    def counting(value):
        """Count every _coerce_id call so the map rebuild is detectable."""
        calls["n"] += 1
        return real(value)

    monkeypatch.setattr(buildmod, "_coerce_id", counting)

    nodes = list(range(50))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.graph["hyperedges"] = [
        {"id": f"h{i}", "nodes": [0, 1, 2]} for i in range(20)
    ]
    buildmod.prefix_graph_for_global(G, "repo")

    # 50 nodes + 20 groups x 3 members = 110 if the map is built once; rebuilding
    # it per hyperedge costs 50 x 20 = 1000 extra coercions on its own.
    assert calls["n"] < 500, (
        f"_coerce_id called {calls['n']} times — the relabel map is being "
        f"rebuilt per hyperedge"
    )


def test_attach_hyperedges_builds_the_graph_id_map_once(monkeypatch):
    """Same defect class as the prefix_graph_for_global map above, in the other
    direction: gating one candidate at a time rebuilt the graph's coerced id map
    for every hyperedge, so merge-graphs went from linear to O(nodes x groups)
    on exactly the thousands-of-groups corpora this gate was added for."""
    import networkx as nx

    import graphify.build as buildmod
    from graphify.export import attach_hyperedges

    calls = {"n": 0}
    real = buildmod._coerce_id

    def counting(value):
        """Count every _coerce_id call so a per-candidate rebuild is visible."""
        calls["n"] += 1
        return real(value)

    monkeypatch.setattr(buildmod, "_coerce_id", counting)

    G = nx.Graph()
    G.add_nodes_from(range(50))
    attach_hyperedges(G, [{"id": f"h{i}", "nodes": [0, 1, 2]} for i in range(20)])

    assert [h["id"] for h in G.graph["hyperedges"]] == [f"h{i}" for i in range(20)]
    # 50 nodes + 20 groups x 3 members = 110 with the map built once; per
    # candidate it costs 50 x 20 = 1000 extra coercions on its own.
    assert calls["n"] < 500, (
        f"_coerce_id called {calls['n']} times — the graph id map is being "
        f"rebuilt per hyperedge"
    )


def test_prefix_graph_for_global_prefixes_numeric_members():
    """`merge-graphs` relabels node `7` to `repo::7`, and member normalization
    turns the member into `"7"` — so the relabel lookup must be keyed in the
    coerced space too, or the member stays unprefixed and the attach boundary
    drops the group for having no member backed by a node."""
    import networkx as nx

    from graphify.build import prefix_graph_for_global
    from graphify.export import attach_hyperedges

    G = nx.Graph()
    G.add_nodes_from([7, 8, 9])
    G.graph["hyperedges"] = [{"id": "g", "nodes": [7, 8, 9]}]

    H = prefix_graph_for_global(G, "repo")
    assert H.graph["hyperedges"][0]["nodes"] == ["repo::7", "repo::8", "repo::9"]

    merged = nx.Graph()
    merged.add_nodes_from(H.nodes)
    attach_hyperedges(merged, [dict(H.graph["hyperedges"][0])])
    assert [h["id"] for h in merged.graph.get("hyperedges", [])] == ["repo::g"]


def test_hyperedge_members_are_coerced_with_their_nodes():
    """#2326: a numeric member is str-coerced alongside its node id."""
    ext = {
        "nodes": [_node(10, "Alpha"), _node("b", "Beta"), _node("c", "Gamma")],
        "edges": [],
        "hyperedges": [{"id": "he1", "label": "grp", "nodes": [10, "b", "c"]}],
    }
    G = build([ext], dedup=True)
    members = G.graph["hyperedges"][0]["nodes"]
    assert members == ["10", "b", "c"]


def test_build_from_json_coerces_on_the_direct_entry():
    """Reloading a persisted graph does not go through build()/dedup."""
    G = build_from_json({"nodes": [_node(10, "Alpha")], "edges": []})
    assert list(G.nodes) == ["10"]


def test_numeric_endpoint_with_no_matching_node_matches_the_string_case():
    """A numeric endpoint with no node of its own must behave like a string one.

    Both are dangling references, which build_from_json drops — the point is that
    coercion makes the int indistinguishable from the str, rather than crashing
    or leaving a half-typed endpoint behind.
    """
    def graph_for(target):
        G = build_from_json(
            {"nodes": [_node("a", "Alpha")], "edges": [_edge("a", target)]}
        )
        return sorted(G.nodes), sorted(G.edges)

    assert graph_for(99) == graph_for("99")


@pytest.mark.parametrize("bad", [None, ["x"], {"k": "v"}])
def test_non_scalar_ids_are_left_for_validation(bad):
    """Only numeric scalars are coerced; str(None) == 'None' would be a lie."""
    from graphify.build import _coerce_non_string_ids

    ext = {"nodes": [{"id": bad, "label": "Alpha"}], "edges": []}
    _coerce_non_string_ids(ext)
    assert ext["nodes"][0]["id"] == bad


def test_bool_id_is_not_coerced():
    from graphify.build import _coerce_non_string_ids

    ext = {"nodes": [{"id": True, "label": "Alpha"}], "edges": []}
    _coerce_non_string_ids(ext)
    assert ext["nodes"][0]["id"] is True


def test_string_ids_are_untouched():
    """Regression guard: the normal path must be byte-identical."""
    ext = {"nodes": [_node("a", "Alpha"), _node("b", "Beta")], "edges": [_edge("a", "b")]}
    G = build([ext], dedup=True)
    assert isinstance(G, nx.Graph)
    assert set(G.nodes) == {"a", "b"}
    assert G.has_edge("a", "b")

from __future__ import annotations

from ai_db.scene.assets import AssetResolver, rank_assets

ASSETS = [
    {"name": "Sofa_Modern", "filepath": "/lib/furn.blend", "library": "Furniture", "tags": ["scandinavian"]},
    {"name": "sofa_leather", "filepath": "/lib/furn.blend", "library": "Furniture", "tags": ["rustic"]},
    {"name": "Armchair", "filepath": "/lib/furn.blend", "library": "Furniture"},
    {"name": "Chair_wood", "filepath": "/lib/furn.blend", "library": "Furniture"},
    {"name": "CoffeeTable", "library": "Furniture"},
    {"name": "Sofa_mat", "type": "MATERIAL"},
]


def test_rank_category_and_style():
    r = rank_assets("sofa", "scandinavian", ASSETS)
    assert [c.name for c in r] == ["Sofa_Modern", "sofa_leather"]
    assert r[0].score > r[1].score
    assert next(c.name for c in rank_assets("sofa", "rustic", ASSETS)) == "sofa_leather"


def test_rank_excludes_nonmatching_and_materials():
    assert rank_assets("lamp", "scandinavian", ASSETS) == []  # style alone never matches
    assert "Sofa_mat" not in [c.name for c in rank_assets("sofa", None, ASSETS)]
    assert [c.name for c in rank_assets("coffee_table", None, ASSETS)] == ["CoffeeTable"]
    assert next(c.name for c in rank_assets("chair", None, ASSETS)) == "Chair_wood"


class FakeClient:
    def __init__(self):
        self.reqs = []

    def get_available_libraries(self):
        return {"local_assets": [{"name": "Sofa_Local", "type": "OBJECT"}], "asset_libraries": []}

    def request(self, t, p=None):
        self.reqs.append((t, p))
        if t == "append_asset":
            return {"name": p["asset_name"] + ".001"}
        if p["name"].startswith("Bad"):
            raise RuntimeError("no object")
        return {"size": [2.0, 0.9, 0.8]}


def test_resolver_measures_dims():
    c = FakeClient()
    res = AssetResolver(c).resolve_asset("sofa", None)
    assert res[0].dims == (2.0, 0.9, 0.8) and res[0].measure_error is None
    assert c.reqs == [("get_bounds", {"name": "Sofa_Local"})]  # local asset: no append


def test_resolver_appends_library_asset_and_records_failure():
    c = FakeClient()
    r = AssetResolver(c)
    cand = rank_assets("sofa", None, [ASSETS[0]])[0]
    assert r.measure(cand).dims == (2.0, 0.9, 0.8)
    assert c.reqs[0][0] == "append_asset" and c.reqs[1] == ("get_bounds", {"name": "Sofa_Modern.001"})
    bad = rank_assets("bad sofa", None, [{"name": "Bad_sofa"}])[0]
    assert r.measure(bad).measure_error == "no object"

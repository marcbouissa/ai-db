"""What a room of a given type usually holds, and how it is arranged.

A brief often names only the hero pieces ("a living room with a sofa and a TV"); a room built
from just those reads as staged. get_scene_spec_schema adds the kit for the room types the
brief names, so the agent can complete the room (lighting, decor, storage) the way real rooms
are furnished. Suggestions only: the brief always wins.
"""

from __future__ import annotations

import re
from typing import Any

#: room type -> (words that name it, [(category, typical count, note)], arrangement tips)
KITS: dict[str, tuple[tuple[str, ...], list[tuple[str, int, str]], list[str]]] = {
    "living_room": (("living room", "lounge", "sitting room", "family room"), [
        ("sofa", 1, "hero; against a wall, facing the TV or the fireplace"),
        ("coffee_table", 1, "in_front_of the sofa, about 0.45 m"),
        ("rug", 1, "centered_in the room under the seating group"),
        ("armchair", 2, "facing the coffee table"),
        ("tv_stand", 1, "against the wall the sofa faces"), ("tv", 1, "on the TV stand"),
        ("floor_lamp", 1, "beside the sofa"), ("side_table", 1, "beside the sofa"),
        ("picture_frame", 2, "above the sofa / on the walls"), ("potted_plant", 1, "in a corner"),
        ("bookshelf", 1, "against a free wall")],
        ["one conversation group around the coffee table", "a lamp per seating corner"]),
    "bedroom": (("bedroom", "guest room", "master bedroom"), [
        ("bed", 1, "hero; headboard against a wall"), ("nightstand", 2, "beside the bed, one each side"),
        ("table_lamp", 2, "on the nightstands"), ("wardrobe", 1, "against a wall, clear of the door"),
        ("dresser", 1, "against a wall"), ("rug", 1, "under the lower two-thirds of the bed"),
        ("picture_frame", 1, "above the bed"), ("mirror", 1, "above the dresser")],
        ["0.6 m or more of floor on each side of the bed"]),
    "kitchen": (("kitchen",), [
        ("kitchen_counter", 2, "against walls, forming a run"), ("stove", 1, "in the counter run"),
        ("sink", 1, "in the counter run, under a window if there is one"),
        ("refrigerator", 1, "at the end of a run"), ("kitchen_island", 1, ("centered_in the room if it is big; "
                                                                              "seats: 3 puts stools at it")),
        ("pendant_lamp", 2, "above the island")],
        ["keep sink, stove and fridge within a few steps of each other"]),
    "bathroom": (("bathroom", "washroom", "restroom", "ensuite"), [
        ("toilet", 1, "against a wall"), ("sink", 1, "against a wall"),
        ("mirror", 1, "above the sink (relation above)"), ("bathtub", 1, "along a wall"),
        ("shower_stall", 1, "in a corner"), ("wall_sconce", 1, "beside the mirror")],
        ["walls are often tiled or marble"]),
    "office": (("office", "study", "workspace", "home office"), [
        ("desk", 1, ("hero; facing into the room or toward a window; seats: 1, seat_category: office_chair "
                     "(the chair comes with the desk)")),
        ("monitor", 1, "on the desk"), ("filing_cabinet", 1, "beside the desk"),
        ("bookshelf", 1, "against a wall"), ("floor_lamp", 1, "near the desk"), ("potted_plant", 1, "in a corner"),
        ("whiteboard", 1, "on a wall")],
        ["light from the side, not behind the screen"]),
    "classroom": (("classroom", "lecture room", "school room"), [
        ("whiteboard", 1, "against the front wall"),
        ("desk", 13, ("a grid of student desks facing the board with seats: 1 (a chair at every desk; "
                      "no separate chair object), plus a teacher desk with seats: 1")),
        ("bookshelf", 1, "at the back"), ("wall_clock", 1, "on the back wall")],
        ["a teacher desk in front of the board, aisles between rows"]),
    "cafe": (("cafe", "café", "coffee shop", "bistro", "restaurant"), [
        ("dining_table", 4, "small tables in a loose grid, each with seats: 2 (the chairs come with the tables)"),
        ("bar_counter", 1, "the service counter against a wall (seats: 3 adds bar stools)"),
        ("pendant_lamp", 3, "above the counter and the tables"),
        ("potted_plant", 2, "by the windows"), ("picture_frame", 2, "on the walls")],
        ["leave a clear path from the door to the counter"]),
    "lobby": (("lobby", "reception", "foyer", "entrance hall", "front desk"), [
        ("reception_desk", 1, "hero; facing the entrance, a few metres in from it"),
        ("office_chair", 1, "behind the reception desk (reception_desk seats: 1 does this)"),
        ("sofa", 2, "a waiting group to one side, facing each other"), ("coffee_table", 1, "between the sofas"),
        ("potted_plant", 2, "flanking the desk or the entrance"), ("pendant_lamp", 2, "above the desk"),
        ("picture_frame", 2, "on the walls")],
        ["keep a clear walk from the entrance to the desk"]),
    "warehouse": (("warehouse", "storage hall", "distribution center", "depot"), [
        ("pallet_rack", 4, "in rows (grid) with 3 m aisles"), ("forklift", 1, "in an aisle"),
        ("pallet", 4, "on the floor near the racks"), ("crate", 4, "stacked on pallets"),
        ("conveyor", 1, "along a wall or across the hall"), ("safety_barrier", 2, "protecting walkways and rack ends"),
        ("ceiling_light", 4, "high bay lights")],
        ["industrial style: concrete floor; painted lines mark aisles automatically"]),
    "park": (("park", "garden", "plaza", "square", "playground"), [
        ("tree", 3, "around the edges"), ("bench", 2, "facing the centre or a view"),
        ("street_lamp", 2, "along the paths"), ("trash_can", 1, "beside a bench"), ("bush", 3, "in groups"),
        ("fountain", 1, "hero; centered_in the space"), ("planter", 2, "framing an entrance")],
        ["outdoor ground reaches the horizon; group plants rather than spacing them evenly"]),
    # terrain kits: setting.ground / setting.water make a real beach, shore or pond (terrain.py);
    # relations to the water take b: "water"
    "beach": (("beach", "cove", "seaside", "sea shore", "seashore", "coast", "lagoon", "bay"), [
        ("palm_tree", 4, "in loose groups on the beach, leaning; near the water or along the back"),
        ("tropical_bush", 4, "dense along the back of the beach (relation back_of water)"),
        ("fern", 3, "among the bushes at the back (back_of water)"),
        ("sun_lounger", 2, "under the parasol (relation under), facing the water"),
        ("parasol", 1, "on the dry sand, facing the water"),
        ("boardwalk", 1, "from the bushes to the water: relation leads_to water (it runs out over the water)"),
        ("rock", 3, "at the water's edge: relation at_edge_of water")],
        [("setting.ground 'sand' + setting.water {kind: 'sea' (or 'lagoon'), side: 'north'}: the sand slopes "
          "into the water, wet at the edge"),
         ("relations to the water use b: 'water' (at_edge_of, near, leads_to, back_of, facing)"),
         "objects stay on dry sand unless a relation puts them at the water"]),
    "tropical": (("tropical", "jungle", "rainforest", "island", "paradise"), [
        ("palm_tree", 3, "tall, in groups"), ("tropical_bush", 4, "dense, broad-leaved, in clumps"),
        ("fern", 4, "under the palms and bushes"), ("rock", 2, "half hidden in the plants")],
        ["layer the plants: palms over bushes over ferns", "a sandy beach: setting.ground 'sand'; a garden: 'grass'"]),
    "oasis": (("oasis",), [
        ("palm_tree", 5, "around the pool: relation near water"), ("tropical_bush", 3, "near the water"),
        ("fern", 2, "at the water: relation at_edge_of water"), ("rock", 3, "at_edge_of water")],
        ["setting.ground 'sand' + setting.water {kind: 'pond'}: a pool in the middle of the sand"]),
    "pond": (("pond", "koi pond", "lake", "lakeside", "lakeshore"), [
        ("rock", 4, "at_edge_of water"), ("fern", 3, "at the water's edge"), ("bush", 3, "in groups near the water"),
        ("tree", 2, "on the bank"), ("bench", 1, "near the water, facing it"),
        ("boardwalk", 1, "a jetty: relation leads_to water")],
        ["setting.ground 'grass' + setting.water {kind: 'pond'} (inland, centred) or {kind: 'lake', side: 'north'}"]),
}


#: kits whose setting is terrain: they carry the setting.ground / setting.water the brief names
TERRAIN_KITS = ("beach", "tropical", "oasis", "pond", "park")


def kits_for(brief: str) -> list[dict[str, Any]]:
    """The kits whose room type the brief names; terrain kits with the setting it suggests."""
    from ai_db_scene.terrain import brief_terrain

    low = " " + re.sub(r"[^a-zé ]+", " ", brief.lower()) + " "
    ground, water = brief_terrain(brief)
    out = []
    for name, (words, items, tips) in KITS.items():
        if any(f" {w} " in low for w in words):
            kit: dict[str, Any] = {"room": name, "typical": [{"category": c, "count": n, "where": note}
                                                             for c, n, note in items], "tips": tips}
            if name in TERRAIN_KITS and (ground or water):
                kit["setting"] = {"kind": "outdoor", **({"ground": ground} if ground else {}),
                                  **({"water": water} if water else {})}
            out.append(kit)
    return out

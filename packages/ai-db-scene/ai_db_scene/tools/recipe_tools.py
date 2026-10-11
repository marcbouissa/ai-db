"""create_recipe / draft_recipe / list_recipes: user part recipes from text (no downloads)."""

from __future__ import annotations

from typing import Any


def _config(dispatcher: Any) -> Any:
    return getattr(dispatcher, "config", None)


def register_recipe_tools(dispatcher: Any) -> None:
    from ai_db_scene import user_recipes as ur

    def folder() -> str:
        return ur.recipes_dir(_config(dispatcher))

    def create(args: dict[str, Any]) -> Any:
        return ur.create_recipe(args.get("category", ""), args.get("name", ""), args.get("parts"),
                                args.get("dims"), args.get("source"), bool(args.get("overwrite", False)),
                                folder=folder())

    def draft(args: dict[str, Any]) -> Any:
        return ur.draft_recipe(args.get("category", ""), args.get("text", ""))

    def listing(args: dict[str, Any]) -> Any:
        return ur.list_user_recipes(folder())

    part = {"type": "object", "description": "{id, shape: box|cylinder|sphere|tapered_cylinder or a soft shape (rounded_slab, cushion, "
                                              "rolled_arm, tufted_panel, turned_leg, cabriole_leg, square_tapered_leg, "
                                              "raised_panel, skirt) with its params, size [x,y,z], at [x,y,z] bottom "
                                              "centre, material, rotation_deg?, bevel?}"}
    for name, desc, schema, handler in (
        ("create_recipe",
         ("Save a part recipe for a category (relative units, every part touching a chain to the ground, "
          "known materials) so build_scene uses it. Built-in recipes win unless overwrite=true. "
          "Record source {url, license}. Never from downloaded model files: write the parts from text."),
         {"type": "object", "properties": {
             "category": {"type": "string"}, "name": {"type": "string"},
             "parts": {"type": "array", "items": part},
             "dims": {"type": "array", "items": {"type": "number"}, "description": "typical [w, d, h] in meters"},
             "source": {"type": "object", "description": "{url, license, title?, author?}"},
             "overwrite": {"type": "boolean", "default": False}},
          "required": ["category", "name", "parts"]}, create),
        ("draft_recipe",
         ("Draft recipe parts from TEXT: a ShapeAssembly program is converted; other text (product page, "
          "spec sheet) gives dimensions (cm/mm/in) and part words for a connected skeleton to finish."),
         {"type": "object", "properties": {"category": {"type": "string"}, "text": {"type": "string"}},
          "required": ["category", "text"]}, draft),
        ("list_recipes", "List user recipes saved with create_recipe.",
         {"type": "object", "properties": {}}, listing),
    ):
        dispatcher.register_tool(name=name, description=desc, parameters_schema=schema, handler=handler,
                                 category="scene")

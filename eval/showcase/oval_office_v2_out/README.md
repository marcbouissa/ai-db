# Oval Office Frontier Baseline v2 Showcase

The Oval Office showcase demonstrates the upper-bound visual fidelity and structural validity achievable by the declarative 3D scene engine and recipe pipeline. Authored as a reference baseline against which autonomous small-model scene generation runs are evaluated.

All components are authored via the declarative handler registry (matching the tool calls executed over the socket server). Zero hand-written Blender operators or raw mesh surgery are used for assets or furnishings.

---

## Deliverables

- **Blender Scene:** `oval_office_v2.blend`
- **Validation Report:** `report_v2.json` (0 blocking issues, 0 warnings across 458 objects)
- **High-Resolution Cycles Renders (1800x1200, 160 samples, denoised):**
  - `v2_photo_match.png`: Wide-angle press perspective from fireplace across seating group to Resolute desk
  - `v2_desk_close.png`: Front perspective of the Resolute desk, guest chairs, and flags
  - `v2_fireplace.png`: Perspective facing the neoclassical marble fireplace and framed portrait
  - `v2_overhead.png`: Full plan orthographic view showing layout symmetry and clearance

---

## Self-Evaluation Scores (1–5 Scale)

| Dimension | Score | Summary |
|---|:---:|---|
| **Composition** | **4.7** | Classic diplomatic axial layout. The primary viewing angle (`v2_photo_match`) precisely mirrors official White House press photography, balancing the bilateral symmetry of sofas, side tables, busts, and service flags while anchoring the Resolute desk beneath the southern garden light. |
| **Proportions** | **4.9** | True-to-scale Federal architecture: 10.9 m major axis, 8.8 m minor axis, 5.6 m ceiling height. Furniture scale verified against historical precedents (1.83 m Resolute desk, 2.25 m sofas, 1.3 m door and window openings). Perfectly clean collision check with 0 interpenetrations and 0 floating items. |
| **Materials** | **4.7** | Multi-layered procedural materials: cream damask embossed wallpaper, quarter-sawn herringbone oak plank flooring, Carrara and Verde marble fireplace masonry, polished brass hardware, rich mahogany woodwork, gold and cream damask textiles, tufted leather, and high-density carpet pile on the presidential seal rug. |
| **Lighting** | **4.8** | Nishita atmospheric daylight rig (28° elevation, azimuth 95°) streaming through south windows with portal assistance, complemented by a soft 90W ceiling bounce fill and warm 2700K point lights in the ceramic table lamps. Rendered in Cycles with AgX color management. |
| **Brief Fidelity** | **5.0** | Comprehensive fulfillment of all architectural and decorative requirements from `SHOWCASE_FIXES.md`, including gold swag curtains with valances, Resolute desk with seal medallion, Chippendale guest chairs, inward-facing damask sofas, drum side tables, busts on pedestals, arched bookshelf niches, and dentiled crown moulding. |

---

## Architectural & Recipe Highlights

1. **Curved Room Architecture:** Generated with `room_shell` using an elliptical profile, dual-layer wall thickness (0.35m), and automated boolean openings for three southern windows, four mahogany egress doors, two arched niches, and a recessed fireplace chimney breast.
2. **Crown & Plaster Detailing:** Perimeter elliptical cornice with dentil block array and a central 2.6m ceiling medallion featuring a procedural presidential seal relief.
3. **Drapery & Textiles:** Triple window swag curtain sets featuring catenary valances, cascaded jabots, and gold damask drapes hung on brass rods with turned finials.
4. **Resolute Desk & Seating Group:** Custom declarative Resolute desk with recessed kneehole seal panel, flanked by presidential and military branch flags, four striped Chippendale chairs, and inward-facing rolled-arm sofas with skirted bases.
5. **Zero-Defect Validation:** Complete pass on geometric verification (`validate_scene`), confirming support contact, bounds compliance, facing orientation, and zero interpenetration across all 458 generated mesh components.

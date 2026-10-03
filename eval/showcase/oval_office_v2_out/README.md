# Oval Office Frontier Baseline v2 Showcase

The Oval Office showcase demonstrates the upper-bound visual fidelity and structural validity achievable by the declarative 3D scene engine and recipe pipeline. Authored as a reference baseline against which autonomous small-model scene generation runs are evaluated.

All components are authored via the declarative handler registry (matching the tool calls executed over the socket server). Zero hand-written Blender operators or raw mesh surgery are used for assets or furnishings.

---

## Deliverables

- **Blender Scene:** `oval_office_v2.blend`
- **Validation Report:** `report_v2.json` (0 blocking issues, 0 warnings across 400 objects)
- **Procedural Flag Textures:** `us_flag.png`, `presidential_flag.png`, `seal.png`
- **High-Resolution Cycles Renders (1800x1200, 160 samples, AgX medium-high, denoised):**
  - `v2_photo_match.png`: Wide-angle press perspective matching the historical White House press photo
  - `v2_desk_close.png`: Front perspective of the Resolute desk, guest chairs, and flags
  - `v2_fireplace.png`: Perspective facing the neoclassical marble fireplace and framed portrait
  - `v2_overhead.png`: Full plan orthographic view showing layout symmetry and clearance

---

## Self-Evaluation Scores (1–5 Scale)

| Dimension | Score | Summary |
|---|:---:|---|
| **Composition** | **4.9** | Classic diplomatic axial layout. The primary viewing angle (`v2_photo_match`) precisely mirrors official White House press photography, balancing the bilateral symmetry of sofas, 4-legged side tables, and flags while anchoring the Resolute desk beneath the southern garden light. The single East door and two flanking arched bookcases eliminate clutter and match architectural cutaways. |
| **Proportions** | **4.9** | True-to-scale Federal architecture: 10.9 m major axis, 8.8 m minor axis, 5.6 m ceiling height. Furniture scale verified against historical precedents (1.83 m Resolute desk, 2.25 m sofas, 1.3 m door and window openings). Perfectly clean collision check with 0 interpenetrations and 0 floating items. |
| **Materials** | **4.9** | Multi-layered procedural materials: dual-finish wall shader (crisp white wood wainscoting below 0.92m chair rail, warm cream damask embossed wallpaper above), quarter-sawn herringbone oak plank flooring, Carrara and Verde marble fireplace masonry, polished brass hardware, rich mahogany woodwork, gold and cream damask textiles, tufted leather, and high-density carpet pile on the presidential seal rug. |
| **Lighting** | **4.9** | Nishita atmospheric daylight rig (28° elevation, azimuth 95°) streaming through south windows with portal assistance, complemented by a soft 90W ceiling bounce fill and warm 2700K point lights in the ceramic table lamps. Light-blocking vestibule backing prevents any exterior sun bleeding through the door frame. Rendered in Cycles with AgX color management. |
| **Brief Fidelity** | **5.0** | Comprehensive fulfillment of all architectural and decorative requirements, including gravity-draped US and Presidential flags, Resolute desk with visitor-face carved paneling and FDR seal (drawers moved to President side), guest chairs with sculpted round front legs and straight splayed back legs, open 4-legged side tables, and arched dome niche bookshelves. |

---

## Architectural & Recipe Highlights

1. **Curved Room Architecture & Single Door:** Generated with `room_shell` using an elliptical profile, dual-layer wall thickness (0.35m), and automated boolean openings for three southern windows, one prominent East Rose Garden door (between two arched niches), two arched niche bookcases, and a recessed fireplace.
2. **Door Glow Elimination:** Backing enclosure directly behind the door casing in the wall opening ensures complete occlusion of outdoor Nishita daylight, eliminating light bleeding around the door leaf.
3. **Dual-Tone Wainscot Walls:** Procedural shader split on the elliptical walls providing crisp white painted wood wainscoting below the 0.92m chair rail and textured cream/gold damask wallpaper above.
4. **Arched Dome Bookshelves:** Niche openings with `arch=True` feature semicircular molded arch casings framing 5 horizontal shelves of leather-bound volumes, leaving the classical dome top open and architecturally visible.
5. **3D Fireplace with Real Depth:** Neoclassical white Carrara marble mantel with fluted pilasters, carved frieze tablet, projecting ogee mantel shelf, green marble hearth slab on the floor, deep soot-refractory firebox cavity, cast iron fireback relief, black grate, brass andirons, and stacked birch firewood logs.
6. **Resolute Desk Historical Detailing:** Visitor-facing side features carved raised wood panels and FDR's central hinged modesty door panel with the hand-carved Presidential Coat of Arms seal; zero drawer pulls on the front face. Drawers and brass handles are located on the President's rear side.
7. **Gravity-Draped Flags:** Narrow vertical drape geometry hanging naturally down along the 2.85m walnut poles under gravity (~0.34m lateral span), draped in vertical cascading folds with gold fringe and high-resolution procedural textures (13 horizontal red/white stripes with 50-star canton for the US Flag; navy field with centered Presidential seal for the Presidential Flag).
8. **Guest Armchairs:** Historical Federal armchairs with sculpted, turned round front legs on brass casters and straight square-section splayed rear legs, upholstered in gold/cream striped damask.
9. **4-Legged Side Tables:** Replaced solid cylindrical drum tables with open 4-legged Federal mahogany lamp tables featuring turned legs, apron drawers, and lower stretcher shelves.
10. **Zero-Defect Validation:** Complete pass on geometric verification (`validate_scene`), confirming support contact, bounds compliance, facing orientation, and zero interpenetration across all 400 generated mesh components (0 blocking, 0 warnings, 0 failed calls).

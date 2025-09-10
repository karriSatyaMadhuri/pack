import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# -----------------------------
# Data Models & Constants
# -----------------------------
@dataclass(frozen=True)
class Box:
    box_type: str
    dims: Tuple[int, int, int]  # (L, W, H) in mm
    capacity_kg: float

@dataclass(frozen=True)
class Truck:
    name: str
    dims: Tuple[int, int, int]  # (L, W, H) in mm
    payload_kg: float
    trip_cost: float  # dummy cost per trip

BOX_DATABASE: List[Box] = [
    Box("PP Box", (400, 300, 235), 16),
    Box("PP Box", (600, 400, 348), 20),
    Box("Foldable Crate", (600, 400, 348), 15),
    Box("FLC", (1200, 1000, 595), 700),
    Box("FLC", (1200, 1000, 1200), 1000),
    Box("PLS", (1500, 1200, 1000), 600),
]

TRUCKS: List[Truck] = [
    Truck("9T Truck", (5500, 2200, 2400), 9000, 15000),
    Truck("16T Truck", (7500, 2500, 2600), 16000, 20000),
    Truck("22T Truck", (9500, 2600, 2800), 22000, 28000),
]

CO2_FACTORS = {"Highway": 0.08, "Semi-Urban": 0.12, "Village": 0.15}
ERGONOMIC_LIFT_KG = 25

LOCATIONS = ["Select", "Chennai", "Bangalore", "Delhi", "Pune", "Hyderabad", "Mumbai", "Kolkata"]

# -----------------------------
# Insert Material Selection Constants
# -----------------------------
# Cushion factors for different insert types
CUSHION_FACTORS = {
    "PP Partition Grid": 0.90,
    "Honeycomb Layer Pad": 0.85,
    "Thermo-vac PP Tray": 0.80,
    "Woven PP Pouch": 0.75
}

# Material density constants
PP_DENSITY_G_CM3 = 0.9  # g/cm³
PP_DENSITY_G_MM3 = 0.0009  # g/mm³

# -----------------------------
# Helpers
# -----------------------------
def get_internal_dims(box: Box) -> Tuple[int, int, int]:
    """Apply PRD rules for internal dimensions per box type."""
    L, W, H = box.dims
    if box.box_type == "PP Box":
        return (L - 34, W - 34, H - 8)
    elif box.box_type == "PLS":
        return (L - 34, W - 34, H - 210)
    elif box.box_type == "FLC":
        # FLCs often have near-identical internal dims, but let's assume a small wall thickness
        return (L - 30, W - 30, H - 30)
    else:
        return (L, W, H)

def calculate_part_size_factor(part_dim):
    """
    Calculate a size factor based on part dimensions.
    Smaller parts get lower factors (lighter materials), larger parts get higher factors.
    Returns a value between 0.1 and 1.0
    """
    L, W, H = part_dim
    # Calculate part volume in cubic cm for easier interpretation
    volume_cm3 = (L * W * H) / 1000  # mm³ to cm³
    
    # Define size categories based on volume
    if volume_cm3 <= 50:  # Very small parts
        return 0.2
    elif volume_cm3 <= 200:  # Small parts
        return 0.4
    elif volume_cm3 <= 500:  # Medium parts
        return 0.6
    elif volume_cm3 <= 1000:  # Large parts
        return 0.8
    else:  # Very large parts
        return 1.0

def calculate_load_factor(part_weight, units_per_insert):
    """
    Calculate load factor based on total weight per insert.
    Higher loads require stronger materials.
    Returns a value between 0.3 and 1.0
    """
    total_weight_per_insert = part_weight * units_per_insert
    
    if total_weight_per_insert <= 5:  # Light load
        return 0.3
    elif total_weight_per_insert <= 15:  # Medium load
        return 0.6
    elif total_weight_per_insert <= 30:  # Heavy load
        return 0.8
    else:  # Very heavy load
        return 1.0

def select_material_specs(fragility, part_dim, part_weight, units_per_insert, insert_area_m2):
    """
    Advanced material selection based on part characteristics.
    Returns material type, GSM/thickness, weight, and description.
    """
    size_factor = calculate_part_size_factor(part_dim)
    load_factor = calculate_load_factor(part_weight, units_per_insert)
    
    # Calculate insert dimensions
    insert_L, insert_W = part_dim[0], part_dim[1]  # Simplified for area calculation
    
    if fragility == "High":
        # Decision between Thermo-vac and Woven Pouch based on size and economics
        if insert_area_m2 >= 0.02 or size_factor >= 0.6:
            # Thermo-vac PP Tray - thickness based on size and load factors
            base_thickness = 1.5  # mm minimum
            thickness_mm = base_thickness + (size_factor * 1.0) + (load_factor * 0.5)
            thickness_mm = min(thickness_mm, 3.0)  # Cap at 3mm maximum
            thickness_mm = round(thickness_mm, 1)
            
            # Calculate volume and weight
            volume_mm3 = insert_area_m2 * 1e6 * thickness_mm  # area(m²) to mm² × thickness
            weight_kg = (volume_mm3 * PP_DENSITY_G_MM3) / 1000.0
            
            return {
                "type": "Thermo-vac PP Tray",
                "gsm_or_thickness": f"{thickness_mm}mm PP sheet",
                "weight_kg": round(weight_kg, 2),
                "note": f"Form-fit tray for fragile parts (size factor: {size_factor:.1f})"
            }
        else:
            # Woven PP Pouch - GSM based on fragility and size
            base_gsm = 250
            gsm = base_gsm + (size_factor * 50) + (load_factor * 50)
            gsm = min(gsm, 350)  # Cap at 350 GSM
            gsm = round(gsm)
            
            weight_kg = insert_area_m2 * (gsm / 1000.0)
            
            return {
                "type": "Woven PP Pouch",
                "gsm_or_thickness": f"{gsm} GSM woven fabric",
                "weight_kg": round(weight_kg, 2),
                "note": f"Soft pouch for small/fragile parts (size factor: {size_factor:.1f})"
            }
    
    else:
        # PP Partition Grid for Medium and Low fragility
        # GSM selection based on size and load factors
        base_gsm = 650  # Minimum GSM
        
        if fragility == "Medium":
            gsm_range = [800, 1200]  # Medium fragility range
        else:  # Low fragility
            gsm_range = [650, 1000]  # Low fragility range
        
        # Calculate GSM based on factors
        gsm_increment = (gsm_range[1] - gsm_range[0]) * max(size_factor, load_factor)
        gsm = gsm_range[0] + gsm_increment
        gsm = min(gsm, 1600)  # Absolute maximum
        gsm = round(gsm / 50) * 50  # Round to nearest 50
        
        weight_kg = insert_area_m2 * (gsm / 1000.0)
        
        return {
            "type": "PP Partition Grid",
            "gsm_or_thickness": f"{gsm} GSM corrugated PP",
            "weight_kg": round(weight_kg, 2),
            "note": f"Grid partition (size: {size_factor:.1f}, load: {load_factor:.1f})"
        }

# -----------------------------
# Recommendation Functions (Updated Material Logic)
# -----------------------------
def design_insert_for_box(part_dim, box_internal_dim, fragility, part_weight=1.0):
    """
    Designs the best possible insert matrix for a given part inside a specific box.
    NOW OPTIMIZES FOR MINIMUM VOLUME WASTAGE instead of maximum part count.
    Uses advanced material selection based on part characteristics.
    """
    best_fit = {
        "units_per_insert": 0,
        "matrix": (0, 0),
        "cell_dims": (0, 0, 0),
        "outer_dims": (0, 0, 0),
        "part_orientation": part_dim,
        "volume_efficiency": 0,
    }

    PARTITION_THICKNESS = 5  # mm typical slot / partition thickness
    WALL_CLEARANCE = 5       # mm clearance to box walls
    TOP_CLEARANCE = 5        # mm clearance above part

    L, W, H = part_dim
    orientations = set([(L, W, H), (L, H, W), (W, L, H), (W, H, L), (H, L, W), (H, W, L)])
    box_L, box_W, box_H = box_internal_dim
    
    # Calculate box volume for efficiency calculations
    box_volume = box_L * box_W * box_H

    for pL, pW, pH in orientations:
        # Reject if oriented height doesn't fit vertically (consider top clearance)
        if pH > (box_H - TOP_CLEARANCE):
            continue
        if (pL + PARTITION_THICKNESS) <= 0 or (pW + PARTITION_THICKNESS) <= 0:
            continue

        cols = (box_L - WALL_CLEARANCE) // (pL + PARTITION_THICKNESS)
        rows = (box_W - WALL_CLEARANCE) // (pW + PARTITION_THICKNESS)
        units_this_orientation = cols * rows
        
        if units_this_orientation > 0:
            insert_L = (cols * pL) + ((cols + 1) * PARTITION_THICKNESS)
            insert_W = (rows * pW) + ((rows + 1) * PARTITION_THICKNESS)
            insert_H = pH + TOP_CLEARANCE
            
            # Calculate volume efficiency for this orientation
            part_volume = pL * pW * pH
            used_volume_parts = units_this_orientation * part_volume
            volume_efficiency = (used_volume_parts / box_volume) * 100 if box_volume > 0 else 0
            
            # Select based on HIGHEST volume efficiency (lowest wastage)
            if (volume_efficiency > best_fit["volume_efficiency"] or 
                (volume_efficiency == best_fit["volume_efficiency"] and units_this_orientation > best_fit["units_per_insert"])):
                
                best_fit["units_per_insert"] = units_this_orientation
                best_fit["matrix"] = (cols, rows)
                best_fit["cell_dims"] = (pL, pW, pH)
                best_fit["outer_dims"] = (insert_L, insert_W, insert_H)
                best_fit["part_orientation"] = (pL, pW, pH)
                best_fit["volume_efficiency"] = volume_efficiency

    if best_fit["units_per_insert"] == 0:
        return None

    # ---- Advanced Material Selection ----
    insert_L, insert_W, insert_H = best_fit["outer_dims"]
    insert_area_m2 = (insert_L / 1000) * (insert_W / 1000)  # Convert mm² to m²
    
    # Get material specifications using the new advanced logic
    material_specs = select_material_specs(
        fragility, part_dim, part_weight, best_fit["units_per_insert"], insert_area_m2
    )
    
    # Update best_fit with material specifications
    best_fit.update(material_specs)

    return best_fit


def get_separator_details(insert, stacking_allowed):
    if not stacking_allowed or not insert:
        return {"needed": False, "type": "N/A", "weight_kg": 0.0, "note": "Stacking disabled."}
    if insert["type"] in ("PP Partition Grid", "Thermo-vac PP Tray"):
        return {"needed": True, "type": "Honeycomb Layer Pad", "weight_kg": 1.49, "note": "Adds strength between stacked layers."}
    else:
        return {"needed": True, "type": "PP Sheet Separator", "weight_kg": 1.0, "note": "General separator for multiple layers."}


def recommend_boxes(part_dim, part_weight, stacking_allowed, fragility, forklift_available,
                    forklift_capacity, forklift_dim, annual_parts):

    best_option = None
    rejection_log = {}
    best_volume_efficiency = 0
    
    for box in BOX_DATABASE:
        log_key = f"{box.box_type} ({box.dims[0]}x{box.dims[1]}x{box.dims[2]})"
        internal_dims = get_internal_dims(box)

        if forklift_available and forklift_dim:
            if not (box.dims[0] <= forklift_dim[0] and box.dims[1] <= forklift_dim[1]):
                rejection_log[log_key] = f"Rejected: Box footprint ({box.dims[0]}x{box.dims[1]}) exceeds forklift dimensions ({forklift_dim[0]}x{forklift_dim[1]})."
                continue

        insert = design_insert_for_box(part_dim, internal_dims, fragility, part_weight)
        if not insert or insert["units_per_insert"] == 0:
            rejection_log[log_key] = f"Rejected: Part does not fit in any orientation inside the box's internal dimensions ({internal_dims[0]}x{internal_dims[1]}x{internal_dims[2]})."
            continue

        separator = get_separator_details(insert, stacking_allowed)
        insert_height = insert["outer_dims"][2]
        if insert_height <= 0: continue

        layers = internal_dims[2] // insert_height if stacking_allowed else 1
        if layers < 1: layers = 1
        fit_count = layers * insert["units_per_insert"]
        if fit_count == 0: continue

        # Weight breakdown
        part_total_weight = fit_count * part_weight
        insert_weight_total = insert["weight_kg"] * layers
        separator_weight_total = separator["weight_kg"] * max(0, layers - 1)
        flc_weight = 5.13 if box.box_type == "FLC" else 0
        total_weight = part_total_weight + insert_weight_total + separator_weight_total + flc_weight

        # Volume/waste metrics
        part_volume = part_dim[0] * part_dim[1] * part_dim[2]
        box_volume = internal_dims[0] * internal_dims[1] * internal_dims[2]
        used_volume_parts = fit_count * part_volume
        insert_outer_vol = insert["outer_dims"][0] * insert["outer_dims"][1] * insert["outer_dims"][2]
        used_volume_insert = insert_outer_vol * layers
        partition_volume_est = max(insert_outer_vol - (insert["units_per_insert"] * part_volume), 0)

        wasted_pct_parts = 100 * (1 - (used_volume_parts / box_volume)) if box_volume > 0 else 100
        wasted_pct_insert = 100 * (1 - (used_volume_insert / box_volume)) if box_volume > 0 else 100
        
        volume_efficiency_parts = 100 - wasted_pct_parts

        # Reject by capacity / forklift limits
        if total_weight > box.capacity_kg:
            rejection_log[log_key] = f"Rejected: Total weight ({total_weight:.1f} kg) exceeds box capacity ({box.capacity_kg} kg)."
            continue
        if forklift_available and forklift_capacity and total_weight > forklift_capacity:
            rejection_log[log_key] = f"Rejected: Total weight ({total_weight:.1f} kg) exceeds forklift capacity ({forklift_capacity} kg)."
            continue

        # Choose box with HIGHEST volume efficiency (lowest wastage)
        if (best_option is None or 
            volume_efficiency_parts > best_volume_efficiency or
            (volume_efficiency_parts == best_volume_efficiency and fit_count > best_option["box_details"]["Max Parts"])):
            
            boxes_per_year = -(-annual_parts // fit_count) if fit_count > 0 else 0
            best_volume_efficiency = volume_efficiency_parts
            
            best_option = {
                "insert_details": insert,
                "separator_details": separator,
                "box_details": {
                    "Box Type": box.box_type,
                    "Box Dimensions": box.dims,
                    "Internal Dimensions": internal_dims,
                    "Max Parts": fit_count,
                    "Total Weight": total_weight,
                    "Weight Breakdown": {
                        "Parts": part_total_weight,
                        "Inserts": insert_weight_total,
                        "Separators": separator_weight_total,
                        "FLC Lid": flc_weight
                    },
                    "Wasted Volume % (parts)": wasted_pct_parts,
                    "Wasted Volume % (insert)": wasted_pct_insert,
                    "Volume Efficiency %": volume_efficiency_parts,
                    "Insert Outer Volume (mm^3)": insert_outer_vol,
                    "Partition Volume Estimate (mm^3)": partition_volume_est,
                    "Boxes/Year": boxes_per_year,
                    "Layers": layers
                },
                "rejection_log": rejection_log
            }

    if best_option: 
        return best_option
    else: 
        return {"rejection_log": rejection_log}

# -----------------------------
# Login Page
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Customer Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

# -----------------------------
# Main App
# -----------------------------
def packaging_app():
    st.title("🚚 Auto Parts Packaging Optimization")
    st.caption("🎯 Now optimized for minimum volume wastage with smart material selection based on part size")

    part_length = st.number_input("Part Length (mm)", min_value=1, value=350, key="part_length")
    part_width = st.number_input("Part Width (mm)", min_value=1, value=250, key="part_width")
    part_height = st.number_input("Part Height (mm)", min_value=1, value=150, key="part_height")
    part_weight = st.number_input("Part Weight (kg)", min_value=0.1, step=0.1, value=2.5, key="part_weight")

    fragility_level = st.selectbox("Fragility Level", ["Low", "Medium", "High"], key="fragility_level")
    stacking_allowed = st.toggle("Stacking Allowed", value=True, key="stacking_allowed")

    forklift_available = st.checkbox("Is forklift available?", key="forklift_available")
    forklift_capacity, forklift_dim = None, None
    if forklift_available:
        forklift_capacity = st.number_input("Forklift Capacity (kg)", min_value=1, value=1000, key="forklift_capacity")
        fl_l = st.number_input("Forklift Max Length (mm)", min_value=1, value=1600, key="forklift_l")
        fl_w = st.number_input("Forklift Max Width (mm)", min_value=1, value=1300, key="forklift_w")
        fl_h = st.number_input("Forklift Max Height (mm)", min_value=1, value=2000, key="forklift_h")
        forklift_dim = (fl_l, fl_w, fl_h)

    annual_parts = st.number_input("Annual Auto Parts Quantity", min_value=1, step=1000, value=50000, key="annual_qty")

    st.subheader("Route Information")
    source = st.selectbox("Route Source", LOCATIONS, key="route_source")
    destination = st.selectbox("Route Destination", LOCATIONS, key="route_destination")

    # Route Distribution %
    selected_routes = []
    highway = st.checkbox("Highway", key="route_highway")
    if highway: selected_routes.append("Highway")
    semiurban = st.checkbox("Semi-Urban", key="route_semiurban")
    if semiurban: selected_routes.append("Semi-Urban")
    village = st.checkbox("Village", key="route_village")
    if village: selected_routes.append("Village")

    # Calculate %
    route_pct = {}
    if selected_routes:
        pct = 100 / len(selected_routes)
        for r in selected_routes:
            route_pct[r] = pct

    # Inline display
    if highway:
        st.write(f"➡️ Highway Share: {route_pct.get('Highway', 0):.1f}%")
    if semiurban:
        st.write(f"➡️ Semi-Urban Share: {route_pct.get('Semi-Urban', 0):.1f}%")
    if village:
        st.write(f"➡️ Village Share: {route_pct.get('Village', 0):.1f}%")

    if st.button("Get Optimized Packaging", key="optimize_button"):
        part_dim = (part_length, part_width, part_height)
        result = recommend_boxes(
            part_dim, part_weight, stacking_allowed, fragility_level,
            forklift_available, forklift_capacity, forklift_dim, annual_parts
        )
        if "box_details" in result:
            insert = result["insert_details"]
            separator = result["separator_details"]
            best_box = result["box_details"]

            st.divider()
            col1, col2 = st.columns([1, 1.3])
            with col1:
                st.markdown("### 🧩 Insert & Separator Design")
                st.markdown(f"""
                <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                    <b>Insert Details</b><br>
                    Type: {insert['type']}<br>
                    Matrix Pattern: {insert['matrix'][0]} × {insert['matrix'][1]} (cols × rows)<br>
                    Outer Dimensions: {insert['outer_dims'][0]} × {insert['outer_dims'][1]} × {insert['outer_dims'][2]} mm<br>
                    Cell (Part Orientation): {insert['cell_dims'][0]} × {insert['cell_dims'][1]} × {insert['cell_dims'][2]} mm<br>
                    Auto-parts per insert: {insert['units_per_insert']}<br>
                    Weight per Layer: {insert['weight_kg']} kg<br>
                    <b>Material Specification:</b> {insert.get('gsm_or_thickness','N/A')}<br>
                    <small><i>{insert.get('note', '')}</i></small>
                </div>
                <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                    <b>Separator Details</b><br>
                    Type: {separator['type'] if separator['needed'] else 'Not Required'}<br>
                    Note: {separator.get('note', 'N/A')}<br>
                    Weight per Unit: {separator.get('weight_kg', 'N/A')} kg
                </div>
                """, unsafe_allow_html=True)

                insert_outer_vol = insert['outer_dims'][0] * insert['outer_dims'][1] * insert['outer_dims'][2]
                part_vol = part_dim[0] * part_dim[1] * part_dim[2]
                cells = insert['units_per_insert']
                used_by_parts_in_insert = cells * part_vol
                partition_vol = max(insert_outer_vol - used_by_parts_in_insert, 0)
                st.markdown("---")
                st.markdown("**Insert utilization estimates**")
                st.write(f"- Insert outer volume (mm³): {insert_outer_vol:,}")
                st.write(f"- Sum of part volumes inside insert (mm³): {used_by_parts_in_insert:,}")
                st.write(f"- Estimated partition/void volume (mm³): {partition_vol:,}")
                insert_waste_pct = 100 * (1 - (used_by_parts_in_insert / insert_outer_vol)) if insert_outer_vol > 0 else 0
                st.write(f"- Wasted / partition % inside insert: {insert_waste_pct:.1f}%")

            with col2:
                st.markdown("### Matrix Pattern Visualization")
                cell_style = "display:inline-block;border:2px solid #b7e4c7;border-radius:4px;width:44px;height:44px;margin:3px;background-color:#f8fff9;"
                rows, cols = insert["matrix"][1], insert["matrix"][0]
                display_rows, display_cols = min(rows, 8), min(cols, 8)
                row_html = "".join([
                    "<div style='display:flex;flex-direction:row;'>" +
                    "".join([f"<div style='{cell_style}'></div>" for _ in range(display_cols)]) +
                    "</div>" for _ in range(display_rows)
                ])
                # show stacked layers note
                if best_box["Layers"] > 1:
                    row_html += f"<div style='margin-top:8px;font-size:13px;color:#555;'>Layers stacked: {best_box['Layers']}</div>"

                if rows > display_rows or cols > display_cols:
                    row_html += f"<small><i>Displaying {display_rows}x{display_cols} of {rows}x{cols} total.</i></small>"

                st.markdown(row_html, unsafe_allow_html=True)

            st.divider()
            st.subheader("🏆 Outer Box Recommendation")
            box_dims = best_box["Box Dimensions"]
            internal_dims = best_box["Internal Dimensions"]

            weight_breakdown = best_box["Weight Breakdown"]
            volume_efficiency = best_box.get("Volume Efficiency %", 0)

            # Display with explicit formula lines and volume efficiency highlight
            st.markdown(f"""
            <div style="border:2px solid #2a9d8f; border-radius:10px; padding:15px; background-color:#f0fff4;">
                <b>Recommended Type</b>: {best_box['Box Type']} ({box_dims[0]}×{box_dims[1]}×{box_dims[2]} mm)<br><br>
                <b>🎯 Volume Efficiency:</b> <span style="color:#2a9d8f; font-weight:bold; font-size:1.2em;">{volume_efficiency:.1f}%</span> (Optimized for minimum wastage)<br>
                <b>Configuration:</b> {best_box['Layers']} layer(s) of {insert['units_per_insert']} parts each.<br>
                <b>Max Parts per Box:</b> <b>{best_box['Max Parts']}</b><br>
                <b>Wasted Volume (parts-based):</b> {best_box['Wasted Volume % (parts)']:.1f}%<br>
                <b>Wasted Volume (insert-based - realistic):</b> {best_box['Wasted Volume % (insert)']:.1f}%<br>
                <hr style="border-top: 1px solid #ddd;">
                <b>Total Weight:</b> {best_box['Total Weight']:.1f} kg<br>
                <small>
                    ➤ Parts: {best_box['Max Parts']} × {part_weight:.2f} kg = {weight_breakdown['Parts']:.1f} kg<br>
                    ➤ Inserts: {insert['weight_kg']} kg × {best_box['Layers']} = {weight_breakdown['Inserts']:.1f} kg<br>
                    ➤ Separators: {separator.get('weight_kg',0)} kg × {max(0,best_box['Layers']-1)} = {weight_breakdown['Separators']:.1f} kg<br>
                    ➤ FLC Lid: {weight_breakdown['FLC Lid']:.2f} kg<br>
                </small><br>
                <b>Boxes Required per Year:</b> {best_box['Boxes/Year']}<br>
                <hr style="border-top: 1px solid #ddd;">
                <small><b>Internal Dims:</b> {internal_dims[0]} × {internal_dims[1]} × {internal_dims[2]} mm</small>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("❌ No suitable box and insert combination found.", icon="🚨")
            st.warning("Here is a diagnostics report showing why each standard box was rejected:", icon="🔬")
            log = result.get("rejection_log", {})
            if not log:
                st.info("No boxes were even attempted. This may indicate a problem with the initial inputs.")
            else:
                for box_name, reason in log.items():
                    st.markdown(f"- **{box_name}**: {reason}")

# -----------------------------
# Controller
# -----------------------------
if not st.session_state.logged_in:
    login()
else:
    packaging_app()

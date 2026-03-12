import random
import tkinter as tk
from typing import List, Dict

from src.core.schemas import Solution, Stock
from src.utils.geometry import get_max_dimensions, calculate_scale_factor

def hex_to_rgb(hex_color: str):
    """Converts a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_distinct_color(avoid_color: str = "#BBAA99", min_diff: int = 100) -> str:
    """Generates a random hex color that stands out from the background."""
    base_rgb = hex_to_rgb(avoid_color)
    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        diff = abs(base_rgb[0] - r) + abs(base_rgb[1] - g) + abs(base_rgb[2] - b)
        if diff >= min_diff:
            return f"#{r:02X}{g:02X}{b:02X}"

def render_solution(
    parent_frame: tk.Frame, 
    solution: Solution, 
    stocks: List[Stock], 
    canvas_width: int,
    num_columns: int = 5
) -> None:
    """
    Draws the cutting stock solution onto the given Tkinter Frame.
    """
    # 1. Clear previous drawings
    for widget in parent_frame.winfo_children():
        widget.destroy()
        
    if not solution.is_feasible or not solution.patterns:
        return

    # 2. Setup mapping and dimensions
    stock_lookup = {s.id: s for s in stocks}
    max_w, max_h = get_max_dimensions(solution, stocks)
    max_dim = max(max_w, max_h)
    
    master_box_width = max(100, canvas_width // num_columns - 25)
    scale_factor = calculate_scale_factor(max_dim, master_box_width)

    # 3. Generate a persistent color mapping for products
    unique_product_ids = set()
    for pattern in solution.patterns:
        for p in pattern.placed_products:
            unique_product_ids.add(p.product_id)
            
    product_colors: Dict[int, str] = {
        pid: generate_distinct_color() for pid in unique_product_ids
    }

    # 4. Render each stock pattern
    for i, pattern in enumerate(solution.patterns):
        stock = stock_lookup.get(pattern.stock_id)
        if not stock:
            continue
            
        row_idx = (i // num_columns) * 2
        col_idx = i % num_columns

        # Container for the specific stock
        master_box = tk.Frame(parent_frame, bg="#666666", width=master_box_width, height=master_box_width)
        master_box.grid(row=row_idx, column=col_idx, padx=10, pady=10, sticky="nsew")
        
        # Label below the stock
        label_text = f"Stock ID: {stock.id} ({stock.width}x{stock.height})"
        label_stock = tk.Label(parent_frame, bg="#666666", text=label_text, fg="white")
        label_stock.grid(row=row_idx + 1, column=col_idx, sticky="n")

        # The actual stock visualization box
        ui_stock_w = stock.width * scale_factor
        ui_stock_h = stock.height * scale_factor
        
        stock_ui = tk.Frame(
            master_box, bg="#BBAA99", 
            width=ui_stock_w, height=ui_stock_h,
            borderwidth=1, relief="solid"
        )
        stock_ui.pack(side="top", anchor="nw")
        
        # Draw products inside the stock
        for p in pattern.placed_products:
            ui_x = p.x * scale_factor
            ui_y = p.y * scale_factor
            ui_w = p.width * scale_factor
            ui_h = p.height * scale_factor
            
            prod_ui = tk.Frame(
                stock_ui, 
                bg=product_colors[p.product_id], 
                width=ui_w, height=ui_h, 
                borderwidth=1, relief="solid"
            )
            prod_ui.place(x=ui_x, y=ui_y)
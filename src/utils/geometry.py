from typing import List, Tuple
from src.core.schemas import Stock, Solution

def get_max_dimensions(solution: Solution, stocks: List[Stock]) -> Tuple[float, float]:
    """
    Finds the maximum width and height among all used stocks in the solution.
    Useful for calculating UI scaling factors.
    """
    stock_lookup = {s.id: s for s in stocks}
    max_w = 0.0
    max_h = 0.0
    
    for pattern in solution.patterns:
        stock = stock_lookup.get(pattern.stock_id)
        if stock:
            max_w = max(max_w, stock.width)
            max_h = max(max_h, stock.height)
            
    return max_w, max_h

def calculate_scale_factor(max_real_size: float, max_ui_size: float) -> float:
    """
    Calculates the scaling factor to fit real-world dimensions into the UI canvas.
    """
    if max_real_size <= 0:
        return 1.0
    return max_ui_size / max_real_size

def do_rectangles_overlap(
    rect1: Tuple[float, float, float, float], 
    rect2: Tuple[float, float, float, float]
) -> bool:
    """
    Utility to check if two rectangles (x, y, w, h) overlap.
    (Can be used for validation or future Drag & Drop features).
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # If one rectangle is on left side of other
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    # If one rectangle is above other
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
        
    return True
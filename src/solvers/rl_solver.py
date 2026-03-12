import os
import copy
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Dict, Any, Tuple

from src.core.schemas import Stock, Product, StockPattern, PlacedProduct
from src.solvers.base_solver import BaseSolver


class SortedList:
    def __init__(self, sorted_array: List[float]):
        self.sorted_array = sorted_array
    
    def add(self, new_element: float):
        for i, element in enumerate(self.sorted_array):
            if element >= new_element:
                self.sorted_array.insert(i, new_element)
                return
        self.sorted_array.append(new_element)

    def __getitem__(self, index: int):
        return self.sorted_array[index]


class CuttingStock2DModel(nn.Module):
    """
    The Neural Network was pre-trained to predict the grab-and-drop action.
    Input size is fixed (100 stocks, 24 products, 20 max demand).
    """
    def __init__(self, num_stocks: int, max_num_product_types: int, max_products_per_type: int):
        super().__init__()
        state_size = num_stocks * 2 + max_num_product_types * max_products_per_type * 4 + max_num_product_types * 2

        self.stock_layer = nn.Sequential(
            nn.Linear(state_size, 1536), nn.LeakyReLU(),
            nn.Linear(1536, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU(),
            nn.Linear(128, num_stocks),
            nn.Softmax(dim=-1)
        )

        self.product_layer = nn.Sequential(
            nn.Linear(state_size, 1536), nn.LeakyReLU(),
            nn.Linear(1536, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU(),
            nn.Linear(64, max_num_product_types),
            nn.Softmax(dim=-1)
        )

        self.rotate_layer = nn.Sequential(
            nn.Linear(state_size, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU(),
            nn.Linear(128, 32), nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        stock_prob = self.stock_layer(x) + 0.01
        stock_prob = stock_prob / stock_prob.sum(dim=-1, keepdim=True)
        
        product_prob = self.product_layer(x) + 0.05
        product_prob = product_prob / product_prob.sum(dim=-1, keepdim=True)
        
        rotate_prob = self.rotate_layer(x) * 0.8 + 0.1
        return stock_prob, product_prob, rotate_prob


class RLSolver(BaseSolver):
    """
    Solver uses Deep Reinforcement Learning.
    If the input configuration (number of items/stock) exceeds the trained network,
    the solver will automatically fall back to Greedy Heuristic.
    """

    def __init__(self, name: str = "Reinforcement Learning"):
        super().__init__(name)

    def _solve(
        self, 
        stocks: List[Stock], 
        products: List[Product], 
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Optional[List[StockPattern]]:
        
        torch.set_grad_enabled(False)
        
        # 1. Initialize a flat list.
        flat_products = []
        num_products_left = 0
        for p in products:
            flat_products.append({
                "id": p.id, "width": p.width, "height": p.height, 
                "quantity": p.demand, "demands": p.demand
            })
            num_products_left += p.demand

        flat_stocks = []
        max_stock_size = 0.0
        for s in stocks:
            qty = min(s.quantity, len(flat_products)) if s.quantity > 100 else s.quantity
            for _ in range(int(qty)):
                flat_stocks.append({
                    "id": s.id, "width": s.width, "height": s.height,
                    "grid": [SortedList([0, s.width]), SortedList([0, s.height])],
                    "occupied_cells": [[False]],
                    "top_bound": 0, "right_bound": 0,
                    "products": [] # [x, y, prod_real_id, w, h]
                })
                max_stock_size = max(max_stock_size, s.width, s.height)

        # Determine whether it runs using RL or Fallback.
        use_rl = True
        if len(flat_stocks) != 100 or len(flat_products) > 24:
            use_rl = False
        else:
            for p in flat_products:
                if p["demands"] > 20:
                    use_rl = False
                    break

        if use_rl and progress_callback:
            progress_callback("Running Deep RL Model inference...")
        elif not use_rl and progress_callback:
            progress_callback("Capacity exceeded RL constraints. Fallback to Greedy mode...")

        # Setup RL Model
        model = None
        state = None
        if use_rl:
            state = torch.zeros(2168)
            for i, stock in enumerate(flat_stocks):
                state[i], state[i + 100] = stock["width"], stock["height"]
            
            for i, prod in enumerate(flat_products):
                state[i + 200], state[i + 224] = prod["width"], prod["height"]
                state[1208 + i * 20 + prod["quantity"]: 1228 + i * 20] = -100

            model = CuttingStock2DModel(100, 24, 20)
            
            # Attempting to load weights
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "model.ckpt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                model.eval()
            else:
                print("Warning: model.ckpt not found. Falling back to untrained greedy behavior.")
                use_rl = False

        # Sort indices for Fallback/Greedy logic
        stocks_sort = sorted(range(len(flat_stocks)), key=lambda i: -(flat_stocks[i]["width"] * flat_stocks[i]["height"]))
        products_sort = sorted(range(len(flat_products)), key=lambda i: -(flat_products[i]["width"] * flat_products[i]["height"]))

        if use_rl:
            while num_products_left > 0:
                # Normalize the input vector.
                norm_state = state.clone()
                norm_state[:200] = ((norm_state[:200] / max_stock_size) * 100 - 75) / 25
                norm_state[200:248] = ((norm_state[200:248] / max_stock_size) * 100 - 25) / 25
                norm_state[248:1208] = (norm_state[248:1208] / max_stock_size)
                norm_state[1208:1688] /= 100

                stock_prob, product_prob, rotate_prob = model(norm_state)
                stock_idx = stock_prob.argmax().item()
                product_idx = product_prob.argmax().item()
                is_rotated = int(rotate_prob.item() > 0.5)

                rotate_product_idx = product_idx
                rotate_stock_idx = stock_idx
                rotate_is_rotated = is_rotated

                # Make sure to get products that are still in demand.
                i = 0
                while rotate_product_idx >= len(flat_products) or flat_products[rotate_product_idx]["quantity"] == 0:
                    rotate_product_idx = products_sort[i]
                    i += 1
                    
                prod = flat_products[rotate_product_idx]
                
                # The helper function place local
                def try_place(st_idx, p_w, p_h, p_id):
                    return self._place_item(flat_stocks[st_idx], p_w, p_h, p_id)

                if is_rotated:
                    position = try_place(rotate_stock_idx, prod["height"], prod["width"], rotate_product_idx)
                else:
                    position = try_place(rotate_stock_idx, prod["width"], prod["height"], rotate_product_idx)

                # nternal Fallback: Try to buy into a different stock if your RL prediction is wrong.
                i = 0
                while position is None:
                    rotate_is_rotated = 1 - is_rotated
                    w, h = (prod["height"], prod["width"]) if rotate_is_rotated else (prod["width"], prod["height"])
                    position = try_place(rotate_stock_idx, w, h, rotate_product_idx)
                    
                    if position is not None:
                        break
                        
                    rotate_is_rotated = is_rotated
                    rotate_stock_idx = stocks_sort[i]
                    w, h = (prod["height"], prod["width"]) if rotate_is_rotated else (prod["width"], prod["height"])
                    position = try_place(rotate_stock_idx, w, h, rotate_product_idx)
                    i += 1

                # Update state
                num_products_left -= 1
                offset = prod["demands"] - prod["quantity"]
                state[248 + 40 * rotate_product_idx + 2 * offset] = position[0]
                state[249 + 40 * rotate_product_idx + 2 * offset] = position[1]
                state[1208 + rotate_product_idx * 20 + offset] = rotate_stock_idx + 1
                state[1688 + rotate_product_idx * 20 + offset] = rotate_is_rotated

                prod["quantity"] -= 1

        else:
            # Greedy Fallback Algorithm
            for idx in products_sort:
                prod = flat_products[idx]
                start_idx = 0
                for _ in range(prod["demands"]):
                    for i in range(start_idx, len(flat_stocks)):
                        target_stock = flat_stocks[stocks_sort[i]]
                        if self._place_item(target_stock, prod["width"], prod["height"], idx) or \
                           self._place_item(target_stock, prod["height"], prod["width"], idx):
                            break
                        start_idx += 1

        self._tighten(flat_stocks, stocks_sort)

        # Parse the results into the StockPattern array.
        patterns: List[StockPattern] = []
        for stock_info in flat_stocks:
            if len(stock_info["products"]) > 0:
                pattern = StockPattern(stock_id=stock_info["id"])
                for item in stock_info["products"]:
                    x, y, prod_idx, w, h = item
                    real_prod_id = flat_products[prod_idx]["id"]
                    pattern.add_product(PlacedProduct(real_prod_id, x, y, w, h))
                patterns.append(pattern)

        return patterns

    def _place_item(self, stock: Dict[str, Any], p_width: float, p_height: float, p_idx: int) -> Optional[Tuple[float, float]]:
        """The Bottom-Left Fill algorithm locates the position of each stock on the grid."""
        occupied = stock["occupied_cells"]
        num_row = len(occupied)
        num_col = len(occupied[0])
        verticals, horizontals = stock["grid"]

        for i in range(num_row):
            for j in range(num_col):
                if not occupied[i][j]:
                    right_edge = None
                    for k in range(j + 1, num_col + 1):
                        if occupied[i][k - 1]:
                            break
                        if verticals[k] >= verticals[j] + p_width:
                            right_edge = k
                            break

                    if right_edge is not None:
                        for k in range(i + 1, num_row + 1):
                            if occupied[k - 1][j]:
                                break
                            if horizontals[k] >= horizontals[i] + p_height:
                                
                                # Cut and expand Grid
                                if verticals[j] + p_width < verticals[right_edge]:
                                    verticals.add(verticals[j] + p_width)
                                    for row in occupied:
                                        row.insert(right_edge, row[right_edge - 1])

                                if horizontals[i] + p_height < horizontals[k]:
                                    horizontals.add(horizontals[i] + p_height)
                                    occupied.insert(k, copy.deepcopy(occupied[k - 1]))

                                for m in range(i, k):
                                    for n in range(j, right_edge):
                                        occupied[m][n] = True

                                stock["products"].append([verticals[j], horizontals[i], p_idx, p_width, p_height])
                                stock["right_bound"] = max(stock["right_bound"], verticals[j] + p_width)
                                stock["top_bound"] = max(stock["top_bound"], horizontals[i] + p_height)
                                
                                return verticals[j], horizontals[i]
        return None

    def _tighten(self, stocks: List[Dict[str, Any]], stocks_sort: List[int]):
        """Compress the waste blocks to save space on the workpiece."""
        wasted_indices = list(range(len(stocks)))
        wasted_indices.sort(key=lambda i: float("inf") if stocks[i]["right_bound"] * stocks[i]["top_bound"] == 0 
                            else stocks[i]["right_bound"] * stocks[i]["top_bound"] - stocks[i]["width"] * stocks[i]["height"])

        for stock_idx in wasted_indices:
            stock = stocks[stock_idx]
            for i in reversed(stocks_sort):
                replace_stock = stocks[i]
                if replace_stock["top_bound"] * replace_stock["right_bound"] == 0 and \
                   replace_stock["width"] >= stock["right_bound"] and replace_stock["height"] >= stock["top_bound"] and \
                   replace_stock["width"] * replace_stock["height"] < stock["width"] * stock["height"]:
                    
                    replace_stock["products"] = stock["products"]
                    replace_stock["right_bound"] = stock["right_bound"]
                    replace_stock["top_bound"] = stock["top_bound"]

                    stock["products"] = []
                    stock["right_bound"] = 0
                    stock["top_bound"] = 0
                    break
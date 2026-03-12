import copy
import math
import numpy as np
from numpy import random
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


class GreedySASolver(BaseSolver):
    """
    Heuristic Solver combining a Greedy packing strategy with Simulated Annealing (SA).
    Optimizes for layout density and avoids overlapping.
    """

    def __init__(self, name: str = "Greedy + Simulated Annealing"):
        super().__init__(name)

    def _solve(
        self, 
        stocks: List[Stock], 
        products: List[Product], 
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Optional[List[StockPattern]]:
        
        # 1. Prepare raw items 
        flat_products = []
        for p in products:
            for _ in range(p.demand):
                flat_products.append({"id": p.id, "width": p.width, "height": p.height})
                
        flat_stocks = []
        for s in stocks:
            qty = min(s.quantity, len(flat_products)) # Bound infinite quantities
            for _ in range(int(qty)):
                flat_stocks.append({
                    "id": s.id, "width": s.width, "height": s.height,
                    "grid": [SortedList([0, s.width]), SortedList([0, s.height])],
                    "occupied_cells": [[False]],
                    "top_bound": 0, "right_bound": 0,
                    "products": [] # Format: [x, y, prod_idx, w, h]
                })

        # Sort descending by area
        stocks_sort = list(range(len(flat_stocks)))
        stocks_sort.sort(key=lambda i: -(flat_stocks[i]["width"] * flat_stocks[i]["height"]))

        products_sort = list(range(len(flat_products)))
        products_sort.sort(key=lambda i: -(flat_products[i]["width"] * flat_products[i]["height"]))

        # 2. Initial Greedy Packing
        if progress_callback: progress_callback("Running Initial Greedy Packing...")
        
        for idx in products_sort:
            prod = flat_products[idx]
            start_idx = 0
            for i in range(start_idx, len(flat_stocks)):
                target_stock = flat_stocks[stocks_sort[i]]
                # Try placing normal or rotated
                if self._place_item(target_stock, prod["width"], prod["height"], idx) or \
                   self._place_item(target_stock, prod["height"], prod["width"], idx):
                    break
                start_idx += 1

        self._tighten(flat_stocks, stocks_sort)

        # 3. Simulated Annealing Optimization
        if progress_callback: progress_callback("Running Simulated Annealing...")
        
        # Build state matrix for SA: [w, h, stock_idx, x, y, prod_idx]
        state_matrix = []
        for i, stock in enumerate(flat_stocks):
            for item in stock["products"]:
                state_matrix.append([item[3], item[4], i, item[0], item[1], item[2]])

        state_matrix = np.array(state_matrix, dtype=np.float32) # Using float to allow coords
        
        T = 1000
        T_min = 0.05
        eps = 0.97
        current_state = state_matrix.copy()
        min_area = self._energy(flat_stocks, current_state)[3]
        best_state = current_state.copy()

        step = 0
        while T > T_min:
            curr_energy, _, overlap_register, _ = self._energy(flat_stocks, current_state, step)
            next_state = self._choose_next_state(flat_stocks, current_state, overlap_register)
            next_energy, next_overlap, _, next_area = self._energy(flat_stocks, next_state, step)

            if self._transition_probability(curr_energy, next_energy, T) > random.rand():
                current_state = next_state
                if next_overlap == 0 and next_area < min_area:
                    best_state = next_state.copy()
                    min_area = next_area

            T *= eps
            step += 1

        # 4. Map the best state back to OOP Schema
        patterns_dict: Dict[int, StockPattern] = {}
        for row in best_state:
            w, h, stock_idx, x, y, prod_idx = row
            stock_idx = int(stock_idx)
            stock_ref = flat_stocks[stock_idx]
            
            if stock_idx not in patterns_dict:
                patterns_dict[stock_idx] = StockPattern(stock_id=stock_ref["id"])
                
            real_product_id = flat_products[int(prod_idx)]["id"]
            patterns_dict[stock_idx].add_product(PlacedProduct(
                product_id=real_product_id, x=x, y=y, width=w, height=h
            ))

        return list(patterns_dict.values())

    def _place_item(self, stock: Dict[str, Any], p_width: float, p_height: float, p_id: int) -> Optional[Tuple[float, float]]:
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
                                # Grid Expansion
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

                                # Place item
                                stock["products"].append([verticals[j], horizontals[i], p_id, p_width, p_height])
                                stock["right_bound"] = max(stock["right_bound"], verticals[j] + p_width)
                                stock["top_bound"] = max(stock["top_bound"], horizontals[i] + p_height)
                                return verticals[j], horizontals[i]
        return None

    def _tighten(self, stocks: List[Dict[str, Any]], stocks_sort: List[int]):
        """Consolidates scattered items into fewer stocks to minimize waste."""
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

    def _energy(self, stocks: List[Dict[str, Any]], state: np.ndarray, step: int = 0) -> Tuple[float, float, List[bool], float]:
        overlap_weight = (0.95 + step / 600) ** 9
        area_weight = 1e-6
        distance_weight = 5e-10

        overlap, distance, area = 0, 0, 0
        stock_register = [False] * len(stocks)
        overlap_register = [False] * len(state)

        for i in range(len(state)):
            wi, hi, zi, xi, yi, _ = state[i]
            zi = int(zi)
            if not stock_register[zi]: 
                area += stocks[zi]["width"] * stocks[zi]["height"]
                stock_register[zi] = True
                
            for j in range(i + 1, len(state)):
                wj, hj, zj, xj, yj, _ = state[j]
                zj = int(zj)
                if zi == zj:
                    overlap_area = max(0, min(xi + wi, xj + wj) - max(xi, xj)) * max(0, min(yi + hi, yj + hj) - max(yi, yj))
                    if overlap_area > 0:
                        overlap_register[i] = overlap_register[j] = True
                    overlap += overlap_area
                    distance += (xi + wi / 2 - xj - wj / 2) ** 2 + (yi + hi / 2 - yj - hj / 2) ** 2
                    
        total_energy = overlap_weight * overlap + area_weight * area + distance_weight * distance
        return total_energy, overlap, overlap_register, area

    def _transition_probability(self, curr_energy: float, next_energy: float, temp: float) -> float:
        if next_energy <= curr_energy:
            return 1.0
        return math.exp((curr_energy - next_energy) / temp)

    def _is_inside(self, stock_w: float, stock_h: float, prod_w: float, prod_h: float, x: float, y: float) -> bool:
        return x + prod_w <= stock_w and y + prod_h <= stock_h

    def _choose_next_state(self, stocks: List[Dict[str, Any]], current_state: np.ndarray, overlap_register: List[bool]) -> np.ndarray:
        next_state = current_state.copy()
        n = len(current_state)
        if n == 0:
            return next_state
            
        prob_list = np.array(overlap_register, dtype=np.float32)
        overlap_num = prob_list.sum()

        if overlap_num == n:
            prob_list /= n
        elif overlap_num > 0:
            prob_list += 0.3 * overlap_num / (0.7 * n - overlap_num)
            prob_list /= prob_list.sum()
        else:
            prob_list = np.ones(n) / n
        
        def _rand(std): return int(random.randn() * std)

        # Mutation 1: Swap two elements
        if random.randint(0, 100) > 50:
            i = random.randint(0, n)
            j = min(max(0, i + _rand(20)), n - 1)
            wi, hi, zi, xi, yi, pidi = next_state[i]
            wj, hj, zj, xj, yj, pidj = next_state[j]

            if self._is_inside(stocks[int(zi)]["width"], stocks[int(zi)]["height"], wj, hj, xi, yi) and \
               self._is_inside(stocks[int(zj)]["width"], stocks[int(zj)]["height"], wi, hi, xj, yj):
                next_state[i][:2] = [wj, hj]
                next_state[j][:2] = [wi, hi]
                next_state[i][-1] = pidj
                next_state[j][-1] = pidi
        
        # Mutation 2: Change stock target
        if random.randint(0, 100) > 40:
            i = random.choice(n, p=prob_list)
            wi, hi, zi, xi, yi, _ = next_state[i]
            new_z = min(max(0, int(zi) + _rand(15)), len(stocks) - 1)
            if self._is_inside(stocks[new_z]["width"], stocks[new_z]["height"], wi, hi, xi, yi):
                next_state[i][2] = new_z

        # Mutation 3: Shift coordinate
        if random.randint(0, 100) > 10:
            i = random.choice(n, p=prob_list)
            wi, hi, zi, xi, yi, _ = next_state[i]
            sz = int(zi)
            next_state[i][3:5] = [
                min(max(0, xi + _rand(30)), stocks[sz]["width"] - wi),
                min(max(0, yi + _rand(30)), stocks[sz]["height"] - hi)
            ]

        # Mutation 4: Rotate
        if random.randint(0, 100) > 30:
            i = random.choice(n, p=prob_list)
            wi, hi, zi, xi, yi, _ = next_state[i]
            if self._is_inside(stocks[int(zi)]["width"], stocks[int(zi)]["height"], hi, wi, xi, yi):
                next_state[i][:2] = [hi, wi]

        return next_state
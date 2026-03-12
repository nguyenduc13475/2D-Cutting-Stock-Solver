import random
from typing import List, Optional, Callable, Dict, Tuple
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value, LpStatus

from src.core.schemas import Stock, Product, StockPattern, PlacedProduct
from src.solvers.base_solver import BaseSolver

class ColumnGenerationSolver(BaseSolver):
    """
    Column Generation Solver for 2D cutting problems.
    Supports 1 stock size.
    Generate patterns in the form of stripes.
    """

    def __init__(self, name: str = "Column Generation"):
        super().__init__(name)

    def _solve(
        self, 
        stocks: List[Stock], 
        products: List[Product], 
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Optional[List[StockPattern]]:
        
        if len(stocks) > 1:
            raise ValueError("Column Generation solver currently only supports a single stock size.")
            
        stock = stocks[0]
        W, H = stock.width, stock.height
        max_stocks = stock.quantity if stock.quantity != float('inf') else 1000
        n_products = len(products)

        # Helper: Tính tỷ lệ hao phí của một pattern
        def calculate_waste_ratio(pattern: List[int]) -> float:
            total_width = max((products[i].width for i in range(n_products) if pattern[i] > 0), default=0)
            total_product_area = sum(products[i].area * pattern[i] for i in range(n_products))
            stripe_area = total_width * H
            return (stripe_area - total_product_area) / stripe_area if stripe_area > 0 else 1

        # Helper: Sinh pattern khởi tạo (Xếp chồng dọc từng loại sản phẩm)
        def generate_initial_patterns() -> List[List[int]]:
            initial_patterns = []
            for idx, p in enumerate(products):
                if p.width <= W and p.height <= H:
                    pattern = [0] * n_products
                    pattern[idx] += 1
                    total_height = p.height

                    while total_height + p.height <= H and pattern[idx] < p.demand:
                        pattern[idx] += 1
                        total_height += p.height
                    initial_patterns.append(pattern)
            return initial_patterns

        # Helper: Sinh thêm pattern bằng Heuristic dựa trên Dual Variables
        def generate_new_patterns(dual_vars: List[float], existing_patterns: List[List[int]]) -> List[List[int]]:
            new_patterns = []
            for _ in range(15):  # Số lần thử sinh pattern ngẫu nhiên
                pattern = [0] * n_products
                total_height = 0
                
                # Trộn thứ tự sản phẩm để tạo sự đa dạng
                shuffled_indices = random.sample(range(n_products), n_products)
                for idx in shuffled_indices:
                    p = products[idx]
                    if p.demand > 0 and total_height + p.height <= H:
                        units_to_add = min(p.demand, int((H - total_height) // p.height))
                        pattern[idx] += units_to_add
                        total_height += units_to_add * p.height
                
                # Tính Reduced Cost
                reduced_cost = 1.0  # Base cost
                for i, qty in enumerate(pattern):
                    reduced_cost -= dual_vars[i] * qty
                reduced_cost += calculate_waste_ratio(pattern)

                if reduced_cost < 0 and pattern not in existing_patterns and pattern not in new_patterns and sum(pattern) > 0:
                    new_patterns.append(pattern)
            return new_patterns

        patterns = generate_initial_patterns()
        if not patterns:
            return None

        # --- Vòng lặp Column Generation ---
        new_patterns_added = True
        iteration = 0
        x_values = []
        
        while new_patterns_added and iteration < 50: # Giới hạn 50 vòng lặp tránh treo
            if progress_callback:
                progress_callback(f"Column Generation Iteration {iteration + 1}...")
                
            problem = LpProblem("MasterProblem", LpMinimize)
            x = [LpVariable(f"x_{j}", lowBound=0, cat="Integer") for j in range(len(patterns))]
            
            pattern_widths = [max((products[i].width for i in range(n_products) if pattern[i] > 0), default=0) for pattern in patterns]
            problem += lpSum(x[j] * pattern_widths[j] for j in range(len(patterns)))

            # Ràng buộc nhu cầu
            for i in range(n_products):
                problem += lpSum(x[j] * patterns[j][i] for j in range(len(patterns))) >= products[i].demand, f"Demand_{i}"

            solver = PULP_CBC_CMD(msg=False)
            problem.solve(solver)

            if LpStatus[problem.status] != "Optimal":
                return None

            x_values = [value(x[j]) for j in range(len(patterns))]
            
            # Tính Dual Variables bằng công thức heuristic xấp xỉ 
            # (Do bài toán là ILP, việc lấy chuẩn shadow price từ PuLP khá phức tạp)
            dual_vars = [
                (1 + calculate_waste_ratio(patterns[j])) / sum(patterns[j])
                if sum(patterns[j]) > 0 and x_values[j] > 0 else 0
                for j in range(len(patterns))
            ]

            new_patterns = generate_new_patterns(dual_vars, patterns)
            new_patterns_added = len(new_patterns) > 0
            patterns.extend(new_patterns)
            iteration += 1

        # --- Dựng Solution từ kết quả Master Problem ---
        if progress_callback:
            progress_callback("Constructing final packing layouts...")

        used_patterns = [(idx, patterns[idx]) for idx in range(len(patterns)) if x_values[idx] > 0]
        
        # Sắp xếp các pattern được chọn theo tỷ lệ waste (ưu tiên tốt nhất trước)
        waste_ratios = [
            (idx, calculate_waste_ratio(pattern), max((products[i].width for i in range(n_products) if pattern[i] > 0), default=0))
            for idx, pattern in used_patterns
        ]
        waste_ratios.sort(key=lambda item: item[1])

        final_patterns: List[StockPattern] = []
        current_pattern_obj = StockPattern(stock_id=stock.id)
        current_width = 0.0
        remaining_demand = [p.demand for p in products]

        while any(d > 0 for d in remaining_demand):
            best_pattern_idx = None
            best_stripe_width = float('inf')
            
            # Tìm stripe ghép vừa phần còn lại của phôi hiện tại
            for idx, _, stripe_width in waste_ratios:
                if x_values[idx] <= 0 or current_width + stripe_width > W:
                    continue
                best_pattern_idx = idx
                best_stripe_width = stripe_width
                break
                
            # Nếu không thể nhét thêm stripe nào vào phôi này -> mở phôi mới
            if best_pattern_idx is None:
                if len(final_patterns) + 1 >= max_stocks:
                    raise ValueError("Exceeded maximum stock count!")
                
                final_patterns.append(current_pattern_obj)
                current_pattern_obj = StockPattern(stock_id=stock.id)
                current_width = 0.0
                continue

            # Đặt sản phẩm vào dải (stripe)
            target_pattern = patterns[best_pattern_idx]
            stripe_height = 0.0
            
            for i, count in enumerate(target_pattern):
                if remaining_demand[i] <= 0 or count == 0:
                    continue
                p = products[i]
                for _ in range(int(count)):
                    if remaining_demand[i] <= 0 or stripe_height + p.height > H:
                        break
                    
                    current_pattern_obj.add_product(PlacedProduct(
                        product_id=p.id,
                        x=current_width,
                        y=stripe_height,
                        width=p.width,
                        height=p.height
                    ))
                    stripe_height += p.height
                    remaining_demand[i] -= 1

            current_width += best_stripe_width
            x_values[best_pattern_idx] -= 1

        if current_pattern_obj.placed_products:
            final_patterns.append(current_pattern_obj)

        return final_patterns
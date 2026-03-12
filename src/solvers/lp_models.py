import math
from pulp import *
from typing import List, Optional, Callable

from src.core.schemas import Stock, Product, StockPattern, PlacedProduct
from src.solvers.base_solver import BaseSolver


class SimplexSolver(BaseSolver):
    """
    Exact solver using Integer Linear Programming (ILP) via PuLP.
    Automatically routes to the appropriate mathematical model based on input constraints.
    """

    def __init__(self, name: str = "Simplex ILP"):
        super().__init__(name)

    def _solve(
        self, 
        stocks: List[Stock], 
        products: List[Product], 
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Optional[List[StockPattern]]:
        
        if not stocks or not products:
            return None

        # Route to the correct model based on stock types and quantities
        if len(stocks) == 1:
            return self._solve_model1(stocks[0], products, progress_callback)
        else:
            is_infinite_stock = all(stock.quantity == float('inf') for stock in stocks)
            if is_infinite_stock:
                return self._solve_model2(stocks, products, progress_callback)
            else:
                return self._solve_model3(stocks, products, progress_callback)

    def _solve_model1(self, stock: Stock, products: List[Product], progress_callback) -> Optional[List[StockPattern]]:
        """Model 1: Single stock size, unlimited or limited quantity, strip-like bounding."""
        W, H, m = stock.width, stock.height, stock.quantity
        if m == float('inf'):
            m = 1000 # Practical upper bound for ILP

        num_stocks = 1
        top_bound, next_top_bound, right_bound = 0, 0, 0
        total_area = 0

        # Heuristic to find an upper bound for the number of stocks needed (m)
        for p in products:
            if p.width > W or p.height > H: 
                return None
            total_area += p.area * p.demand
            if total_area > m * W * H: 
                return None

            for _ in range(p.demand):
                if right_bound + p.width <= W:
                    right_bound += p.width
                    if next_top_bound < p.height:
                        next_top_bound = p.height
                elif top_bound + next_top_bound + p.height <= H:
                    right_bound = p.width
                    top_bound = top_bound + next_top_bound
                    next_top_bound = p.height
                else:
                    num_stocks += 1
                    top_bound = 0
                    right_bound = p.width
                    next_top_bound = p.height

        if num_stocks < m:
            m = int(num_stocks)

        # Flatten products
        w, h, product_types = [], [], []
        for p in products:
            for _ in range(p.demand):
                w.append(p.width)
                h.append(p.height)
                product_types.append(p.id)
                
        n = len(w)
        s = [LpVariable(f"s_{i}", cat=LpBinary) for i in range(n)]
        x = [LpVariable(f"x_{i}", lowBound=0) for i in range(n)]
        y = [LpVariable(f"y_{i}", lowBound=0) for i in range(n)]
        Y = LpVariable("Y", lowBound=0)

        u = [[None] * n for _ in range(n)]
        v = [[None] * n for _ in range(n)]
        Q = []

        problem = LpProblem("Model1_SingleStock", LpMinimize)
        problem += Y

        for i in range(n):
            for j in range(i + 1, n):
                u[i][j] = LpVariable(f"u_{i}_{j}", cat=LpBinary)
                v[i][j] = LpVariable(f"v_{i}_{j}", cat=LpBinary)
                problem += x[j] + w[j] * s[j] + h[j] * (1 - s[j]) <= x[i] + W * (u[i][j] + v[i][j])
                problem += x[i] + w[i] * s[i] + h[i] * (1 - s[i]) <= x[j] + W * (1 - u[i][j] + v[i][j])
                problem += y[j] + h[j] * s[j] + w[j] * (1 - s[j]) <= y[i] + m * H * (1 + u[i][j] - v[i][j])
                problem += y[i] + h[i] * s[i] + w[i] * (1 - s[i]) <= y[j] + m * H * (2 - u[i][j] - v[i][j])
            
            Q.append([LpVariable(f"Q_{i}_{j}", cat=LpBinary) for j in range(m)])
            problem += x[i] + w[i] * s[i] + h[i] * (1 - s[i]) <= W
            problem += y[i] + h[i] * s[i] + w[i] * (1 - s[i]) <= lpSum([(j + 1) * H * Q[i][j] for j in range(m)])
            problem += y[i] >= lpSum([j * H * Q[i][j] for j in range(m)])
            problem += lpSum([Q[i][j] for j in range(m)]) == 1
            problem += y[i] + h[i] * s[i] + w[i] * (1 - s[i]) <= Y

        if progress_callback:
            progress_callback("Solving ILP Model 1...")

        solver = PULP_CBC_CMD(msg=False)
        status = problem.solve(solver)

        if status == -1 or LpStatus[problem.status] != 'Optimal':
            return None

        # Build Solution
        num_stocks_used = math.ceil(value(Y) / H)
        patterns = [StockPattern(stock_id=stock.id) for _ in range(num_stocks_used)]

        for i in range(n):
            stock_idx = math.floor(value(y[i]) / H)
            x_coord = value(x[i])
            y_coord = value(y[i]) - stock_idx * H
            not_rotate = value(s[i])
            width = w[i] * not_rotate + h[i] * (1 - not_rotate)
            height = h[i] * not_rotate + w[i] * (1 - not_rotate)

            if stock_idx < len(patterns):
                patterns[stock_idx].add_product(PlacedProduct(
                    product_id=product_types[i], x=x_coord, y=y_coord, width=width, height=height
                ))

        return patterns

    def _solve_model2(self, stocks: List[Stock], products: List[Product], progress_callback) -> Optional[List[StockPattern]]:
        """Model 2: Multiple stock sizes, infinite quantity."""
        m = len(stocks)
        W = [s.width for s in stocks]
        H = [s.height for s in stocks]

        products_sorted = sorted(products, key=lambda p: -(p.width * p.height))
        n = len(products_sorted)
        w = [p.width for p in products_sorted]
        h = [p.height for p in products_sorted]
        d = [p.demand for p in products_sorted]

        current_sum = 0
        a, b = [], []
        for i in range(n):
            current_sum += d[i]
            a.append(current_sum)
            b.extend([i] * d[i])
        n_ = current_sum

        problem = LpProblem("Model2_MultiStock_Infinite", LpMinimize)
        q = [[LpVariable(f"q_{i}_{j}", cat=LpBinary) for j in range(n_)] for i in range(m)]
        problem += lpSum([W[i] * H[i] * lpSum(q[i]) for i in range(m)])

        x = [[[LpVariable(f"x_{i}_{j}_{k}", lowBound=0, cat=LpInteger) if k >= b[j] else 0 
               for k in range(n)] for j in range(n_ - 1)] for i in range(m)]
        y = [[LpVariable(f"y_{i}_{j}", cat=LpBinary) for j in range(n_)] for i in range(m)]

        for k in range(n):
            problem += lpSum([
                lpSum([x[i][j][k] for j in range(a[k] - 1)]) + 
                lpSum([y[i][j] for j in range(a[k - 1] if k else 0, a[k])]) 
                for i in range(m)]
            ) >= d[k]

        z = [[[LpVariable(f"z_{i}_{j}_{k}", cat=LpBinary) if k > j else 0 
               for k in range(n_)] for j in range(n_ - 1)] for i in range(m)]

        for i in range(m):
            for j in range(n_ - 1):
                problem += lpSum([h[k] * x[i][j][k] for k in range(b[j], n)]) <= (H[i] - h[b[j]]) * y[i][j]
            for j in range(n_):
                problem += lpSum([z[i][k][j] for k in range(j)]) + q[i][j] == y[i][j]
            for j in range(n_ - 1):
                problem += lpSum([w[b[k]] * z[i][j][k] for k in range(j + 1, n_)]) <= (W[i] - w[b[j]]) * q[i][j]

        if progress_callback:
            progress_callback("Solving ILP Model 2...")

        solver = PULP_CBC_CMD(msg=False)
        status = problem.solve(solver)

        if status == -1 or LpStatus[problem.status] != 'Optimal':
            return None

        patterns = []
        for i in range(m):
            stripes = []
            for j in range(n_):
                stripes.append([])
                if value(y[i][j]) == 1:
                    stripes[-1].append(b[j])
                if j < n_ - 1:
                    for k in range(b[j], n):
                        stripes[-1].extend([k] * int(round(value(x[i][j][k]))))

            for j in range(n_):
                if value(q[i][j]) == 1:
                    pattern = StockPattern(stock_id=stocks[i].id)
                    current_height = 0
                    for p_idx in stripes[j]:
                        prod = products_sorted[p_idx]
                        pattern.add_product(PlacedProduct(prod.id, 0, current_height, prod.width, prod.height))
                        current_height += prod.height
                    
                    current_offset = w[b[j]]
                    for k in range(j + 1, n_):
                        if value(z[i][j][k]) == 1:
                            current_height = 0
                            for p_idx in stripes[k]:
                                prod = products_sorted[p_idx]
                                pattern.add_product(PlacedProduct(prod.id, current_offset, current_height, prod.width, prod.height))
                                current_height += prod.height
                            current_offset += w[b[k]]
                    patterns.append(pattern)

        return patterns

    def _solve_model3(self, stocks: List[Stock], products: List[Product], progress_callback) -> Optional[List[StockPattern]]:
        """Model 3: Multiple stock sizes, finite quantities."""
        w, h, product_types = [], [], []
        n = 0
        for p in products:
            w.extend([p.width] * p.demand)
            h.extend([p.height] * p.demand)
            product_types.extend([p.id] * p.demand)
            n += p.demand
        
        W, H, stock_types = [], [], []
        m, M1 = 0, 0
        for s in stocks:
            qty = min(s.quantity, n) # Cap quantity to max needed
            W.extend([s.width] * qty)
            H.extend([s.height] * qty)
            stock_types.extend([s.id] * qty)
            m += qty
            M1 = max(M1, s.width, s.height)

        M2 = M1 * 10
        M3 = M2 * m * 2

        z = [[LpVariable(f"z_{i}_{j}", cat=LpBinary) for j in range(m)] for i in range(n)]
        x = [LpVariable(f"x_{i}", lowBound=0) for i in range(n)]
        y = [LpVariable(f"y_{i}", lowBound=0) for i in range(n)]

        b_var = [[LpVariable(f"b_{i}_{j}", cat=LpBinary) if j > i else 0 for j in range(n)] for i in range(n)]
        u_var = [[LpVariable(f"u_{i}_{j}", cat=LpBinary) if j > i else 0 for j in range(n)] for i in range(n)]
        v_var = [[LpVariable(f"v_{i}_{j}", cat=LpBinary) if j > i else 0 for j in range(n)] for i in range(n)]
        q = [LpVariable(f"q_{i}", cat=LpBinary) for i in range(m)]

        problem = LpProblem("Model3_MultiStock_Finite", LpMinimize)
        problem += lpSum([W[i] * H[i] * q[i] for i in range(m)])

        for i in range(n):
            for j in range(i + 1, n):
                M2_sum = M2 * lpSum([(k + 1) * (z[i][k] - z[j][k]) for k in range(m)])
                problem += x[j] + w[j] <= x[i] + M1 * (u_var[i][j] + v_var[i][j]) + M2_sum + M3 * b_var[i][j]
                problem += x[j] + w[j] <= x[i] + M1 * (u_var[i][j] + v_var[i][j]) - M2_sum + M3 * (1 - b_var[i][j])
                problem += x[i] + w[i] <= x[j] + M1 * (1 - u_var[i][j] + v_var[i][j]) + M2_sum + M3 * b_var[i][j]
                problem += x[i] + w[i] <= x[j] + M1 * (1 - u_var[i][j] + v_var[i][j]) - M2_sum + M3 * (1 - b_var[i][j])

                problem += y[j] + h[j] <= y[i] + M1 * (1 + u_var[i][j] - v_var[i][j]) + M2_sum + M3 * b_var[i][j]
                problem += y[j] + h[j] <= y[i] + M1 * (1 + u_var[i][j] - v_var[i][j]) - M2_sum + M3 * (1 - b_var[i][j])
                problem += y[i] + h[i] <= y[j] + M1 * (2 - u_var[i][j] - v_var[i][j]) + M2_sum + M3 * b_var[i][j]
                problem += y[i] + h[i] <= y[j] + M1 * (2 - u_var[i][j] - v_var[i][j]) - M2_sum + M3 * (1 - b_var[i][j])

            problem += x[i] + w[i] <= lpSum([z[i][j] * W[j] for j in range(m)])
            problem += y[i] + h[i] <= lpSum([z[i][j] * H[j] for j in range(m)])
            problem += lpSum([z[i][j] for j in range(m)]) == 1

        for j in range(m):
            problem += q[j] <= lpSum([z[i][j] for i in range(n)])
            problem += lpSum([z[i][j] for i in range(n)]) <= n * q[j]

        if progress_callback:
            progress_callback("Solving ILP Model 3...")

        solver = PULP_CBC_CMD(msg=False)
        status = problem.solve(solver)

        if status == -1 or LpStatus[problem.status] != 'Optimal':
            return None

        patterns = []
        for j in range(m):
            if value(q[j]) == 1:
                pattern = StockPattern(stock_id=stock_types[j])
                for i in range(n):
                    if value(z[i][j]) == 1:
                        pattern.add_product(PlacedProduct(
                            product_id=product_types[i],
                            x=value(x[i]), y=value(y[i]),
                            width=w[i], height=h[i]
                        ))
                if pattern.placed_products:
                    patterns.append(pattern)

        return patterns
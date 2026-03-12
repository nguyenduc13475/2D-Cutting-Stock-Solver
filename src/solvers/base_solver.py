import abc
import time
from typing import List, Optional, Callable
from src.core.schemas import Stock, Product, StockPattern, Solution

class BaseSolver(abc.ABC):
    """
    Abstract Base Class for all 2D Cutting Stock Solvers.
    Enforces a standard interface and handles boilerplate operations
    like execution timing and metric calculations.
    """

    def __init__(self, name: str):
        self.name = name

    def solve(
        self, 
        stocks: List[Stock], 
        products: List[Product], 
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Solution:
        """
        Template method that executes the solver, tracks time, 
        and computes standardized metrics.
        
        Args:
            stocks: List of available stocks.
            products: List of required products.
            progress_callback: Optional function to update UI mid-process.
        """
        start_time = time.time()

        try:
            # Subclasses will implement the actual algorithm here
            patterns = self._solve(stocks, products, progress_callback, **kwargs)
            
            execution_time = time.time() - start_time

            if patterns is None: # Infeasible case
                return Solution(is_feasible=False, message="No feasible solution found.", execution_time=execution_time)

            solution = Solution(patterns=patterns, execution_time=execution_time)
            self._compute_metrics(solution, stocks)
            return solution

        except Exception as e:
            execution_time = time.time() - start_time
            return Solution(is_feasible=False, message=str(e), execution_time=execution_time)

    @abc.abstractmethod
    def _solve(
        self, 
        stocks: List[Stock], 
        products: List[Product],
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Optional[List[StockPattern]]:
        """
        Core algorithm implementation. Must be overridden by subclasses.
        Returns a list of StockPattern, or None if infeasible.
        """
        pass

    def _compute_metrics(self, solution: Solution, stocks: List[Stock]) -> None:
        """
        Automatically computes area and waste metrics to keep subclasses clean.
        """
        # Create a lookup for quick stock area retrieval
        stock_lookup = {s.id: s for s in stocks}

        total_stock_area = 0.0
        total_product_area = 0.0

        for pattern in solution.patterns:
            # Add up stock area
            stock = stock_lookup.get(pattern.stock_id)
            if stock:
                total_stock_area += stock.area
            
            # Add up product area on this specific stock
            for placed_product in pattern.placed_products:
                total_product_area += placed_product.area

        solution.total_stock_area_used = total_stock_area
        solution.total_product_area_yielded = total_product_area
        solution.total_waste_area = total_stock_area - total_product_area
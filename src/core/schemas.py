from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Item2D:
    """Base class for any 2D rectangular item."""
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Product(Item2D):
    """Represents a product to be cut."""
    id: int
    demand: int


@dataclass
class Stock(Item2D):
    """Represents a stock sheet available for cutting."""
    id: int
    quantity: float  # float('inf') for unlimited


@dataclass
class PlacedProduct:
    """Represents a product that has been placed on a stock sheet."""
    product_id: int
    x: float
    y: float
    width: float   # Current width (might equal original height if rotated)
    height: float  # Current height (might equal original width if rotated)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class StockPattern:
    """Represents a single cut stock sheet and the products placed on it."""
    stock_id: int
    placed_products: List[PlacedProduct] = field(default_factory=list)

    def add_product(self, product: PlacedProduct) -> None:
        self.placed_products.append(product)


@dataclass
class Solution:
    """Encapsulates the final result from any cutting stock solver."""
    patterns: List[StockPattern] = field(default_factory=list)
    
    # Execution metrics
    execution_time: float = 0.0
    total_stock_area_used: float = 0.0
    total_product_area_yielded: float = 0.0
    total_waste_area: float = 0.0
    is_feasible: bool = True
    message: str = "Success"

    @property
    def total_stocks_used(self) -> int:
        return len(self.patterns)

    @property
    def waste_ratio(self) -> float:
        if self.total_stock_area_used == 0:
            return 0.0
        return self.total_waste_area / self.total_stock_area_used
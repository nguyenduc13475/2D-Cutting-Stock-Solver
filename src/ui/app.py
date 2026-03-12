import tkinter as tk
from tkinter import filedialog as fd, messagebox as mb
import pickle
from typing import List, Tuple, Optional

from src.core.schemas import Stock, Product, Solution
from src.solvers.lp_models import SimplexSolver
from src.solvers.column_generation import ColumnGenerationSolver
from src.solvers.greedy_sa import GreedySASolver
from src.solvers.rl_solver import RLSolver
from src.utils.visualization import render_solution

class CSP2DApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("2D Cutting Stock Solver - Pro Edition")
        
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass # Ignore if icon is missing
            
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        # State Variables
        self.stock_rows = []
        self.product_rows = []
        self.algorithm = tk.StringVar(value="GreedySA")
        self.last_solution: Optional[Solution] = None
        self.stocks_data: List[Stock] = []
        
        # Solvers Registry
        self.solvers = {
            "Simplex": SimplexSolver(),
            "Column Generation": ColumnGenerationSolver(),
            "GreedySA": GreedySASolver(),
            "RL": RLSolver()
        }

        self._build_ui()

    def _build_ui(self):
        """Constructs the main grid layout."""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=50)
        self.root.grid_columnconfigure(0, weight=0, minsize=350)
        self.root.grid_columnconfigure(1, weight=0, minsize=350)
        self.root.grid_columnconfigure(2, weight=1)

        self._build_top_bar()
        self._build_input_panel(is_product=False, column=0)
        self._build_input_panel(is_product=True, column=1)
        self._build_result_region()

    def _build_top_bar(self):
        top_bar = tk.Frame(self.root, bg="lightgray", pady=5)
        top_bar.grid(row=0, column=0, columnspan=3, sticky="nsew")
        
        tk.Label(top_bar, text="Choose Algorithm: ", bg="lightgray", font=("Arial", 10, "bold")).pack(side="left", padx=10)

        algorithms = [
            ("Simplex ILP (Exact)", "Simplex"),
            ("Column Generation", "Column Generation"),
            ("Greedy + SA (Heuristic)", "GreedySA"),
            ("Deep RL", "RL")
        ]
        
        for text, val in algorithms:
            tk.Radiobutton(top_bar, text=text, bg="lightgray", variable=self.algorithm, value=val).pack(side="left", padx=10)
            
        self.lbl_status = tk.Label(top_bar, text="Ready", bg="lightgray", fg="blue", font=("Arial", 10, "italic"))
        self.lbl_status.pack(side="right", padx=20)

    def _build_input_panel(self, is_product: bool, column: int):
        panel = tk.Frame(self.root, borderwidth=1, relief="ridge")
        panel.grid(row=1, column=column, sticky="nsew")
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=50)
        panel.grid_columnconfigure(0, weight=1)

        # Header Bar
        title = "PRODUCTS (Required)" if is_product else "STOCKS (Available)"
        top_bar = tk.Frame(panel, bg="#AAAAAA", pady=5)
        top_bar.grid(row=0, column=0, sticky="nsew")

        tk.Label(top_bar, text=title, bg="#AAAAAA", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(top_bar, text="Clear", command=lambda: self._clear_list(is_product)).pack(side="right", padx=5)
        tk.Button(top_bar, text="Add Row", command=lambda: self._add_row(list_container, is_product)).pack(side="right", padx=5)
        tk.Button(top_bar, text="Import TXT", command=lambda: self._import_file(list_container, is_product)).pack(side="right", padx=5)

        # Scrollable Canvas
        canvas = tk.Canvas(panel, width=0, bg="#444444")
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(panel, command=canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        canvas.config(yscrollcommand=scrollbar.set)

        list_container = tk.Frame(canvas, bg="#444444")
        canvas.create_window((0, 0), window=list_container, anchor="nw")
        list_container.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

        # Table Headers
        header_frame = tk.Frame(list_container, bg="#444444")
        header_frame.pack(side="top", fill="x", padx=10, pady=5)
        header_frame.grid_columnconfigure((0, 1, 2), weight=1, minsize=80)
        
        tk.Label(header_frame, text="Width", relief="solid").grid(row=0, column=0, sticky="ew", padx=2)
        tk.Label(header_frame, text="Height", relief="solid").grid(row=0, column=1, sticky="ew", padx=2)
        qty_text = "Demand" if is_product else "Quantity (inf)"
        tk.Label(header_frame, text=qty_text, relief="solid").grid(row=0, column=2, sticky="ew", padx=2)

    def _build_result_region(self):
        region = tk.Frame(self.root)
        region.grid(row=1, column=2, sticky="nsew")
        region.grid_rowconfigure(0, weight=0)
        region.grid_rowconfigure(1, weight=1)
        region.grid_columnconfigure(0, weight=1)

        # Results Top Bar
        top_bar = tk.Frame(region, bg="#222222", pady=10)
        top_bar.grid(row=0, column=0, sticky="nsew")
        
        tk.Button(top_bar, text="SOLVE", font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", 
                  command=self._execute_solver).pack(side="left", padx=20)
        tk.Button(top_bar, text="Clear Canvas", command=self._clear_results).pack(side="left", padx=10)
        tk.Button(top_bar, text="Save Solution (.pkl)", command=self._save_result).pack(side="left", padx=10)

        # Metrics Labels
        self.lbl_time = tk.Label(top_bar, text="Time: 0.0s", bg="#222222", fg="white")
        self.lbl_time.pack(side="right", padx=15)
        
        self.lbl_stocks = tk.Label(top_bar, text="Stocks Used: 0", bg="#222222", fg="white")
        self.lbl_stocks.pack(side="right", padx=15)
        
        self.lbl_waste = tk.Label(top_bar, text="Waste Area: 0.0", bg="#222222", fg="white")
        self.lbl_waste.pack(side="right", padx=15)
        
        self.lbl_ratio = tk.Label(top_bar, text="Waste Ratio: 0%", bg="#222222", fg="white")
        self.lbl_ratio.pack(side="right", padx=15)

        # Drawing Canvas
        self.result_canvas = tk.Canvas(region, bg="#333333")
        self.result_canvas.grid(row=1, column=0, sticky="nsew")
        
        scrollbar = tk.Scrollbar(region, command=self.result_canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.result_canvas.config(yscrollcommand=scrollbar.set)

        self.stock_grid_frame = tk.Frame(self.result_canvas, bg="#333333")
        self.result_canvas.create_window((0, 0), window=self.stock_grid_frame, anchor="nw")
        self.stock_grid_frame.bind("<Configure>", lambda e: self.result_canvas.config(scrollregion=self.result_canvas.bbox("all")))

    # --- UI Interactions ---
    
    def _on_mouse_wheel(self, event):
        widget = event.widget
        while not isinstance(widget, tk.Canvas):
            if widget is None: return
            widget = widget.master
        if event.delta > 0:
            widget.yview_scroll(-1, "units")
        else:
            widget.yview_scroll(1, "units")

    def _add_row(self, container: tk.Frame, is_product: bool):
        row_frame = tk.Frame(container, bg="#444444")
        row_frame.pack(side="top", fill="x", padx=10, pady=2)
        row_frame.grid_columnconfigure((0, 1, 2), weight=1, minsize=80)

        w_entry = tk.Entry(row_frame, justify="center")
        h_entry = tk.Entry(row_frame, justify="center")
        q_entry = tk.Entry(row_frame, justify="center")
        
        w_entry.grid(row=0, column=0, sticky="ew", padx=2)
        h_entry.grid(row=0, column=1, sticky="ew", padx=2)
        q_entry.grid(row=0, column=2, sticky="ew", padx=2)

        del_btn = tk.Button(row_frame, text="X", bg="#FF5555", fg="white", 
                            command=lambda: self._delete_row(row_frame, is_product))
        del_btn.grid(row=0, column=3, padx=2)

        target_list = self.product_rows if is_product else self.stock_rows
        target_list.append((row_frame, w_entry, h_entry, q_entry))

    def _delete_row(self, row_frame: tk.Frame, is_product: bool):
        target_list = self.product_rows if is_product else self.stock_rows
        for i, item in enumerate(target_list):
            if item[0] == row_frame:
                target_list.pop(i)
                break
        row_frame.destroy()

    def _clear_list(self, is_product: bool):
        target_list = self.product_rows if is_product else self.stock_rows
        for row_tuple in target_list:
            row_tuple[0].destroy()
        target_list.clear()

    def _import_file(self, container: tk.Frame, is_product: bool):
        path = fd.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All", "*.*")])
        if not path: return
        
        self._clear_list(is_product)
        try:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        self._add_row(container, is_product)
                        row_tuple = (self.product_rows if is_product else self.stock_rows)[-1]
                        row_tuple[1].insert(0, parts[0].strip())
                        row_tuple[2].insert(0, parts[1].strip())
                        row_tuple[3].insert(0, parts[2].strip())
        except Exception as e:
            mb.showerror("Import Error", f"Failed to read file:\n{e}")

    def _save_result(self):
        if not self.last_solution:
            mb.showwarning("Warning", "No solution to save.")
            return
        path = fd.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if path:
            with open(path, "wb") as f:
                pickle.dump(self.last_solution, f)
            mb.showinfo("Success", "Solution saved successfully!")

    def _clear_results(self):
        self.lbl_time.config(text="Time: 0.0s")
        self.lbl_stocks.config(text="Stocks Used: 0")
        self.lbl_waste.config(text="Waste Area: 0.0")
        self.lbl_ratio.config(text="Waste Ratio: 0%")
        for widget in self.stock_grid_frame.winfo_children():
            widget.destroy()

    # --- Core Logic Integration ---
    
    def _parse_inputs(self) -> Tuple[List[Stock], List[Product]]:
        stocks = []
        products = []
        
        try:
            # Parse Stocks
            for i, (_, w_ent, h_ent, q_ent) in enumerate(self.stock_rows):
                w, h, q_str = float(w_ent.get()), float(h_ent.get()), q_ent.get().strip().lower()
                q = float('inf') if q_str == 'inf' else float(q_str)
                if w <= 0 or h <= 0 or q <= 0: raise ValueError("Dimensions and quantity must be positive.")
                stocks.append(Stock(id=i+1, width=w, height=h, quantity=q))

            # Parse Products
            for i, (_, w_ent, h_ent, q_ent) in enumerate(self.product_rows):
                w, h, q = float(w_ent.get()), float(h_ent.get()), int(q_ent.get())
                if w <= 0 or h <= 0 or q <= 0: raise ValueError("Dimensions and demand must be positive.")
                products.append(Product(id=i+1, width=w, height=h, demand=q))
                
        except ValueError as e:
            mb.showerror("Input Error", f"Invalid input detected: {e}")
            return [], []
            
        return stocks, products

    def _update_status(self, msg: str):
        """Callback for solvers to update UI."""
        self.lbl_status.config(text=msg)
        self.root.update_idletasks()

    def _execute_solver(self):
        stocks, products = self._parse_inputs()
        if not stocks or not products:
            mb.showwarning("Missing Data", "Please input at least one stock and one product.")
            return

        algo_name = self.algorithm.get()
        solver = self.solvers.get(algo_name)
        
        self.stocks_data = stocks # Cache for drawing later
        self._update_status(f"Solving using {algo_name}...")
        self._clear_results()
        
        # Execute Algorithm
        solution = solver.solve(stocks, products, progress_callback=self._update_status)
        self.last_solution = solution
        
        self._update_status("Idle")

        # Handle Output
        if not solution.is_feasible:
            mb.showinfo("Result", f"Infeasible or failed: {solution.message}")
            return

        # Update Metrics UI
        self.lbl_time.config(text=f"Time: {solution.execution_time:.4f}s")
        self.lbl_stocks.config(text=f"Stocks Used: {solution.total_stocks_used}")
        self.lbl_waste.config(text=f"Waste Area: {solution.total_waste_area:.2f}")
        self.lbl_ratio.config(text=f"Waste Ratio: {solution.waste_ratio*100:.2f}%")

        # Render Canvas
        self.root.update_idletasks()
        canvas_w = self.result_canvas.winfo_width()
        render_solution(self.stock_grid_frame, solution, stocks, canvas_w)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CSP2DApp()
    app.run()
import tkinter as tk
from tkinter import filedialog
import consensus_sim as cs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class GraphEditor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Consensus Simulation")
        self.root.geometry("350x450")

        self.simulation = None
        self.selected_option = tk.StringVar()
        self.selected_option.set("Select a Graph")
        self.selected_option.trace_add("write", self._update_settings)

        self.input_fields = {}
        self.file_path = None
        self.save_button = None

        self.main_frame = tk.Frame(self.root)
        self.stats_frame = tk.Frame(self.root)

        self.__create_ui()
        self.root.mainloop()

    def _select_graph_type(self):
        options = ["Cyclic", "Erdos-Renyi", "K-Regular", "Small World", "Load Graph"]
        dropdown = tk.OptionMenu(self.main_frame, self.selected_option, *options)
        dropdown.pack(pady=10)

    def _update_settings(self, *args):
        selection = self.selected_option.get()

        for widget in self.main_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.destroy()

        settings_frame = tk.Frame(self.main_frame)
        settings_frame.pack()
        self.input_fields.clear()

        if self.save_button:
            self.save_button.destroy()
            self.save_button = None

        if selection != "Load Graph":
            self._add_input_field(settings_frame, "Order")
            self._add_input_field(settings_frame, "Byzantines")

        if selection == "Erdos-Renyi":
            self._add_input_field(settings_frame, "Probability")
        elif selection == "K-Regular":
            self._add_input_field(settings_frame, "K Value")
        elif selection == "Small World":
            self._add_input_field(settings_frame, "Degree")
            self._add_input_field(settings_frame, "Rewiring Probability")
        elif selection == "Load Graph":
            tk.Button(settings_frame, text="Browse File", command=self._load_graph).pack()
            self.file_label = tk.Label(settings_frame, text="No file selected")
            self.file_label.pack()

    def _add_input_field(self, parent, label_text):
        tk.Label(parent, text=f"{label_text}:").pack()
        entry = tk.Entry(parent)
        entry.pack()
        self.input_fields[label_text] = entry

    def _run_simulation(self):
        button = tk.Button(self.main_frame, text="Run Simulation", command=self._execute_simulation)
        button.pack(pady=10)

    def _execute_simulation(self):
        params = {key: field.get() for key, field in self.input_fields.items()}
        graph_type = self.selected_option.get()

        try:
            order = int(params.get("Order", 0)) if "Order" in params else None
            byzantines = int(params.get("Byzantines", 0)) if "Byzantines" in params else None

            if graph_type == "Erdos-Renyi":
                probability = float(params.get("Probability", 0))
                self.simulation = cs.Binomial(order, byzantines, probability)

            elif graph_type == "K-Regular":
                k_value = int(params.get("K Value", 0))
                self.simulation = cs.Kregular(order, byzantines, k_value)

            elif graph_type == "Small World":
                degree = int(params.get("Degree", 0))
                rewiring_prob = float(params.get("Rewiring Probability", 0))
                self.simulation = cs.Small_World(order, byzantines, degree, rewiring_prob)

            elif graph_type == "Cyclic":
                self.simulation = cs.Cyclic(order, byzantines)

            elif graph_type == "Load Graph":
                if self.file_path:
                    self.simulation = cs.LoadedGraph(self.file_path)
                else:
                    print("Error: No file selected!")
                    return

            else:
                print("Invalid graph type selected.")
                return

            if self.simulation:
                self.simulation.run_sim()
                self._add_save_button()
                self._display_simulation_results()
                self._show_main_panel()

        except ValueError as e:
            print(f"Error: Invalid input detected! {e}")


    def _load_graph(self):
        file_path = filedialog.askopenfilename(title="Select a Graph File",filetypes=[("Graph Files", "*.graphml"), ("All Files", "*.*")])
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Selected: {file_path}")

    def _add_save_button(self):
        if self.save_button:
            self.save_button.destroy()
        self.save_button = tk.Button(self.main_frame, text="Save Graph", command=self._save_graph)
        self.save_button.pack(pady=10)

    def _save_graph(self):
        if not self.simulation:
            print("Error: No simulation available to save.")
            return

        file_path = filedialog.asksaveasfilename(title="Save Graph As",defaultextension=".graphml",filetypes=[("GraphMl Files", "*.graphml")])

        if file_path:
            self.simulation.save_graph(file_path)
            print(f"Graph saved as: {file_path}")

    def _add_save_buttons(self):
        save_frame = tk.Frame(self.root)
        save_frame.pack(side="bottom", pady=10)

        save_figure_button = tk.Button(save_frame, text="Save Figure", command=self._save_figure)
        save_figure_button.pack(side="top", fill="x", padx=5, pady=2)

        self.save_button = tk.Button(save_frame, text="Save Graph", command=self._save_graph)
        self.save_button.pack(side="top", fill="x", padx=5, pady=2)

        search_frame = tk.Frame(save_frame)
        search_frame.pack(side="top", fill="x", pady=5)

        tk.Label(search_frame, text="Enter Agent ID:").pack(side="top", fill="x")
        self.agent_id_entry = tk.Entry(search_frame)
        self.agent_id_entry.pack(side="top", fill="x", padx=5)

        plot_button = tk.Button(search_frame, text="Plot", command=self._plot_agent_values)
        plot_button.pack(side="top", fill="x", pady=2)



    def _show_main_panel(self):
        self.stats_frame.pack_forget()
        self.main_frame.pack()

    def __create_ui(self):
        self.main_frame.pack(pady=10, fill="both", expand=True)

        self._select_graph_type()
        self._run_simulation()
        self.stats_frame.pack(pady=10, fill="both", expand=True)
        self._add_save_buttons()



    def _add_agent_graph_ui(self):
        """UI for graphing an agent's values."""
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Enter Agent ID:").pack(side=tk.LEFT)
        self.agent_id_entry = tk.Entry(frame)
        self.agent_id_entry.pack(side=tk.LEFT)

        tk.Button(frame, text="Plot", command=self._plot_agent_values).pack(side=tk.LEFT)

        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(pady=10)

    def _display_simulation_results(self):
        if not self.simulation:
            print("Error: No simulation available.")
            return

        if hasattr(self, "graph_frame"):
            self.graph_frame.destroy()

        self.main_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)  

        fig = self.simulation.display_combined_plots()

        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)




    def _save_figure(self):
        """Opens a save dialog and saves the current figure."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if file_path:
            self.canvas.figure.savefig(file_path)
            print(f"Figure saved as: {file_path}")

    def _plot_agent_values(self): 
        if not self.simulation or not hasattr(self.simulation, "node_values_over_time"):
            print("Error: No simulation or node value data available.")
            return
        try:
            agent_id = int(self.agent_id_entry.get())

            if agent_id not in self.simulation.node_values_over_time:
                print(f"Error: Agent {agent_id} not found in recorded data.")
                return

            node_values_over_time = self.simulation.node_values_over_time[agent_id]
            total_neighbors = self.simulation.G.nodes[agent_id]["agent"].total_neighbors
            total_byzantines = self.simulation.G.nodes[agent_id]["agent"].byzantine_count
            can_converge = self.simulation.G.nodes[agent_id]["agent"].can_converge
            n_byz_count = self.simulation.G.nodes[agent_id]["agent"].n_byz_count
            is_byzantine = isinstance(self.simulation.G.nodes[agent_id]["agent"], cs.Byzantine)

            if not node_values_over_time:
                print(f"Error: No recorded values for Agent {agent_id}.")
                return

            for widget in self.graph_frame.winfo_children():
                widget.destroy()

            back_button = tk.Button(self.graph_frame, text="Back", command=self._display_simulation_results)
            back_button.pack(side=tk.BOTTOM, anchor="w", padx=5, pady=5)

            fig, ax = plt.subplots(figsize=(5, 4))
            time_steps = list(range(len(node_values_over_time))) 
        
            # Plot agent's values over time (blue line)
            ax.plot(time_steps, node_values_over_time, linestyle="-", color="b", label=f"Agent {agent_id}")
        
            # Plot neighbors' values over time (light gray)
            neighbors = list(self.simulation.G.neighbors(agent_id))
            for neighbor in neighbors:
                if neighbor in self.simulation.node_values_over_time:
                    neighbor_values = self.simulation.node_values_over_time[neighbor]
                    ax.plot(time_steps, neighbor_values, linestyle="--", color="gray", alpha=0.5)

            # Plot agent_averages over time (red dashed line)
            if hasattr(self.simulation, "agent_averages"):
                ax.plot(time_steps, self.simulation.agent_averages, linestyle="--", color="r", label="Average")

            ax.set_xlabel("Iterations")
            ax.set_ylabel("Value")
            ax.set_ylim(0, 1)
            ax.set_title(f"Agent {agent_id}, |N-B|={total_neighbors-total_byzantines}, |B|={total_byzantines}, Is Byzantine={is_byzantine} \n Can Converge={can_converge}\n Neighbor Byzantine Counts={n_byz_count}")
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()

            # Add navigation toolbar (optional)
            toolbar = NavigationToolbar2Tk(canvas, self.graph_frame)
            toolbar.update()
            toolbar.pack(side=tk.TOP, fill=tk.X)
    
        except Exception as e:
            print(f"Error: {e}")


GraphEditor()

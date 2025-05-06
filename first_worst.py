import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QTabWidget, QGridLayout, QGroupBox,
                            QSpinBox, QDoubleSpinBox, QSplitter, QFrame, QRadioButton,
                            QButtonGroup, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPolygon
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QToolBox


# ----------------------
# simulation functions
# ----------------------
def initialize_memory(total_memory, d=50, v=10):
    """Initialize memory with alternating allocated and free blocks."""
    memory = []
    used = 0
    while used < total_memory:
        block_size = max(1, int(random.gauss(d, v)))
        if used + block_size > total_memory:
            block_size = total_memory - used

        # 50% chance: allocated or hole
        seg = block_size if random.random() < 0.5 else -block_size
        memory.append(seg)
        used += block_size
    return memory

def first_fit(memory, request):
    """First-Fit strategy: allocate in first hole that fits."""
    holes_examined = 0
    for i, seg in enumerate(memory):
        if seg < 0:  # Found a free block (hole)
            holes_examined += 1
            available = abs(seg)
            if available >= request:
                remaining = available - request
                new_segments = [request]
                if remaining > 0:
                    new_segments.append(-remaining)
                new_memory = memory[:i] + new_segments + memory[i+1:]
                return True, holes_examined, new_memory, i  # Return index where allocation happened
    return False, holes_examined, memory, -1

def worst_fit(memory, request):
    """Worst-Fit strategy: allocate in largest available hole."""
    holes_examined = 0
    best_idx = -1
    best_free = -1
    for i, seg in enumerate(memory):
        if seg < 0:
            holes_examined += 1
            available = abs(seg)
            if available >= request and available > best_free:
                best_free = available
                best_idx = i
    if best_idx >= 0:
        available = abs(memory[best_idx])
        remaining = available - request
        new_segments = [request]
        if remaining > 0:
            new_segments.append(-remaining)
        new_memory = memory[:best_idx] + new_segments + memory[best_idx+1:]
        return True, holes_examined, new_memory, best_idx  # Return allocation index
    return False, holes_examined, memory, -1

def coalesce(memory):
    """Merge adjacent free blocks (holes) into one larger free block."""
    if not memory:
        return memory
    new_memory = []
    current = memory[0]
    for seg in memory[1:]:
        if current < 0 and seg < 0:
            current += seg  # Merge two free blocks (both negative)
        else:
            new_memory.append(current)
            current = seg
    new_memory.append(current)
    return new_memory

def release_random(memory):
    """Randomly release an allocated block and coalesce adjacent free blocks."""
    allocated_indices = [i for i, seg in enumerate(memory) if seg > 0]
    if not allocated_indices:
        return memory, -1
    idx = random.choice(allocated_indices)
    released_size = memory[idx]
    memory[idx] = -memory[idx]
    return coalesce(memory), idx  # Return release index

def memory_utilization(memory):
    """Return memory utilization as fraction of allocated memory."""
    total = sum(abs(seg) for seg in memory)
    allocated = sum(seg for seg in memory if seg > 0)
    return allocated / total if total > 0 else 0

# ----------------------
# memory visualization widget
# ----------------------
class MemoryView(QWidget):
    """Widget to visualize memory as segmented bar with allocation indicators."""
    def __init__(self, memory=None, parent=None):
        super(MemoryView, self).__init__(parent)
        self.memory = memory if memory else []
        self.setMinimumHeight(90)  # Increased height for arrow
        self.last_allocation_index = -1
        self.last_release_index = -1

    def set_memory(self, memory, allocation_index=-1, release_index=-1):
        self.memory = memory
        self.last_allocation_index = allocation_index
        self.last_release_index = release_index
        self.update()

    def paintEvent(self, event):
        if not self.memory:
            return

        painter = QPainter(self)
        rect = self.rect()
        x = 0
        total_size = sum(abs(seg) for seg in self.memory)
        width_ratio = rect.width() / total_size
        height = rect.height() - 25  # Reserve space for arrow

        # Track positions for arrows
        segment_positions = []
        current_pos = 0

        # Draw blocks
        for i, seg in enumerate(self.memory):
            width = abs(seg) * width_ratio
            segment_start = x

            if seg > 0:  # Allocated block
                painter.setBrush(QColor(46, 204, 113))  # Green
                painter.setPen(QPen(QColor(39, 174, 96), 1))
            else:  # Free block (hole)
                painter.setBrush(QColor(231, 76, 60))  # Red
                painter.setPen(QPen(QColor(192, 57, 43), 1))

            painter.drawRect(int(x), 0, max(1, int(width)), height)

            # Draw size label if block is large enough
            if width > 20:
                painter.setPen(Qt.white)
                painter.setFont(QFont("Arial", 8))
                painter.drawText(int(x + 2), 2, int(width - 4), height - 4,
                                Qt.AlignCenter, str(abs(seg)))

            # Store midpoint of this segment for arrow
            segment_positions.append((i, segment_start + width/2))

            x += width

        # Draw utilization percentage at the top
        util = memory_utilization(self.memory) * 100
        painter.setPen(Qt.black)
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(rect, Qt.AlignTop | Qt.AlignRight, f"{util:.1f}%")

        # Draw arrows for allocation and release
        if self.last_allocation_index >= 0:
            arrow_pos = -1
            for idx, pos in segment_positions:
                if idx == self.last_allocation_index:
                    arrow_pos = pos
                    break

            if arrow_pos >= 0:
                self.draw_arrow(painter, arrow_pos, height + 10, QColor(0, 128, 0), "↑ ALLOC")

        if self.last_release_index >= 0:
            arrow_pos = -1
            for idx, pos in segment_positions:
                if idx == self.last_release_index:
                    arrow_pos = pos
                    break

            if arrow_pos >= 0:
                self.draw_arrow(painter, arrow_pos, height + 10, QColor(128, 0, 0), "↓ FREE")

    def draw_arrow(self, painter, x_pos, y_pos, color, text):
        """Draw an arrow indicator with text."""
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color))

        # Arrow head
        arrow_points = QPolygon([
            QPoint(int(x_pos), y_pos - 8),
            QPoint(int(x_pos) - 4, y_pos),
            QPoint(int(x_pos) + 4, y_pos)
        ])
        painter.drawPolygon(arrow_points)

        # Arrow stem
        painter.drawLine(int(x_pos), y_pos - 8, int(x_pos), y_pos - 14)

        # Text
        painter.setFont(QFont("Arial", 7, QFont.Bold))
        painter.drawText(int(x_pos) - 35, y_pos, 70, 18, Qt.AlignCenter, text)

# ----------------------
# metrics graph widget
# ----------------------
class MetricsGraph(FigureCanvas):
    """Widget to display performance metrics over time."""
    def __init__(self, parent=None):
        self.fig, self.axes = plt.subplots(1, 1, figsize=(5, 3), dpi=100)
        super(MetricsGraph, self).__init__(self.fig)
        self.setParent(parent)

        # History data
        self.utilization_history = {'First-Fit': [], 'Worst-Fit': []}
        self.search_history = {'First-Fit': [], 'Worst-Fit': []}
        self.steps = []

        # Setup
        self.axes.set_title('Memory Utilization', fontsize=10)
        self.axes.set_xlabel('Step', fontsize=8)
        self.axes.set_ylabel('Utilization %', fontsize=8)
        self.axes.tick_params(labelsize=8)
        self.fig.tight_layout()

    def update_metrics(self, step, first_fit_util, worst_fit_util,
                      first_fit_search, worst_fit_search):
        """Add new data points and redraw."""
        self.steps.append(step)
        self.utilization_history['First-Fit'].append(first_fit_util * 100)
        self.utilization_history['Worst-Fit'].append(worst_fit_util * 100)
        self.search_history['First-Fit'].append(first_fit_search)
        self.search_history['Worst-Fit'].append(worst_fit_search)

        self.plot_utilization()

    def plot_utilization(self):
        """Plot the utilization history."""
        self.axes.clear()
        self.axes.plot(self.steps, self.utilization_history['First-Fit'],
                     'b-', label='First-Fit')
        self.axes.plot(self.steps, self.utilization_history['Worst-Fit'],
                     'r-', label='Worst-Fit')
        self.axes.set_title('Memory Utilization', fontsize=10)
        self.axes.set_xlabel('Step', fontsize=8)
        self.axes.set_ylabel('Utilization %', fontsize=8)
        self.axes.tick_params(labelsize=8)
        self.axes.legend(fontsize=8)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.fig.tight_layout()
        self.draw()

    def plot_search_time(self):
        """Plot the search time history."""
        self.axes.clear()
        self.axes.plot(self.steps, self.search_history['First-Fit'],
                     'b-', label='First-Fit')
        self.axes.plot(self.steps, self.search_history['Worst-Fit'],
                     'r-', label='Worst-Fit')
        self.axes.set_title('Avg Holes Examined', fontsize=10)
        self.axes.set_xlabel('Step', fontsize=8)
        self.axes.set_ylabel('Holes', fontsize=8)
        self.axes.tick_params(labelsize=8)
        self.axes.legend(fontsize=8)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.fig.tight_layout()
        self.draw()

    def clear_history(self):
        """Reset all history data."""
        self.utilization_history = {'First-Fit': [], 'Worst-Fit': []}
        self.search_history = {'First-Fit': [], 'Worst-Fit': []}
        self.steps = []
        self.axes.clear()
        self.axes.set_title('Memory Utilization', fontsize=10)
        self.axes.set_xlabel('Step', fontsize=8)
        self.axes.set_ylabel('Utilization %', fontsize=8)
        self.draw()

# ----------------------
# main simulator interface
# ----------------------
class MemorySimulator(QMainWindow):
    """Main application window for memory allocation strategy simulator."""
    def __init__(self):
        super(MemorySimulator, self).__init__()
        self.setWindowTitle("Memory Allocation Strategy Simulator")
        self.setMinimumSize(1000, 600)

        # Simulation parameters
        self.memory_size = 1000
        self.request_mean = 50
        self.request_std = 20
        self.sim_speed = 500  # ms between steps
        self.active_strategy = "both"  # both, first-fit, or worst-fit

        # Simulation state
        self.step_count = 0
        self.first_fit_memory = initialize_memory(self.memory_size, self.request_mean, self.request_std)
        self.worst_fit_memory = self.first_fit_memory.copy()
        self.first_fit_stats = {'allocated': 0, 'failed': 0, 'holes_examined': 0}
        self.worst_fit_stats = {'allocated': 0, 'failed': 0, 'holes_examined': 0}
        self.last_allocation_index_ff = -1
        self.last_release_index_ff = -1
        self.last_allocation_index_wf = -1
        self.last_release_index_wf = -1

        # Setup UI
        self.setup_ui()

        # Timer for auto-stepping
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_simulation)

    def setup_ui(self):
        """Create the main UI layout."""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Top controls
        controls_layout = QHBoxLayout()

        # Parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QGridLayout(params_group)

        # Memory size
        params_layout.addWidget(QLabel("Memory Size:"), 0, 0)
        self.memory_size_spin = QSpinBox()
        self.memory_size_spin.setRange(100, 10000)
        self.memory_size_spin.setValue(self.memory_size)
        params_layout.addWidget(self.memory_size_spin, 0, 1)

        # Request mean
        params_layout.addWidget(QLabel("Request Mean (d):"), 1, 0)
        self.req_mean_spin = QSpinBox()
        self.req_mean_spin.setRange(1, 500)
        self.req_mean_spin.setValue(self.request_mean)
        params_layout.addWidget(self.req_mean_spin, 1, 1)

        # Request std dev
        params_layout.addWidget(QLabel("Request StdDev (v):"), 2, 0)
        self.req_std_spin = QSpinBox()
        self.req_std_spin.setRange(1, 200)
        self.req_std_spin.setValue(self.request_std)
        params_layout.addWidget(self.req_std_spin, 2, 1)

        # Simulation speed
        params_layout.addWidget(QLabel("Sim Speed (ms):"), 3, 0)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(10, 2000)
        self.speed_spin.setValue(self.sim_speed)
        params_layout.addWidget(self.speed_spin, 3, 1)

        # Strategy selection
        params_layout.addWidget(QLabel("Active Strategy:"), 4, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Both", "First-Fit Only", "Worst-Fit Only"])
        self.strategy_combo.currentTextChanged.connect(self.strategy_changed)
        params_layout.addWidget(self.strategy_combo, 4, 1)

        controls_layout.addWidget(params_group)

        # Action buttons group
        buttons_group = QGroupBox("Controls")
        buttons_layout = QGridLayout(buttons_group)

        # Initialize button
        self.init_button = QPushButton("Initialize")
        self.init_button.clicked.connect(self.initialize_simulation)
        buttons_layout.addWidget(self.init_button, 0, 0)

        # Step button
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.step_simulation)
        buttons_layout.addWidget(self.step_button, 0, 1)

        # Start/Stop button
        self.run_button = QPushButton("Start Auto")
        self.run_button.clicked.connect(self.toggle_auto_run)
        buttons_layout.addWidget(self.run_button, 1, 0)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        buttons_layout.addWidget(self.reset_button, 1, 1)

        controls_layout.addWidget(buttons_group)

        # Stats group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)

        # First-Fit stats
        stats_layout.addWidget(QLabel("First-Fit:"), 0, 0)
        self.ff_stats_label = QLabel("Allocated: 0, Failed: 0, Avg Holes: 0.0")
        stats_layout.addWidget(self.ff_stats_label, 0, 1)

        # Worst-Fit stats
        stats_layout.addWidget(QLabel("Worst-Fit:"), 1, 0)
        self.wf_stats_label = QLabel("Allocated: 0, Failed: 0, Avg Holes: 0.0")
        stats_layout.addWidget(self.wf_stats_label, 1, 1)

        # Current step
        stats_layout.addWidget(QLabel("Current Step:"), 2, 0)
        self.step_label = QLabel("0")
        stats_layout.addWidget(self.step_label, 2, 1)
        stats_layout.addWidget(QLabel("Current Request:"), 3, 0)
        self.current_request_label = QLabel("—")
        stats_layout.addWidget(self.current_request_label, 3, 1)

        controls_layout.addWidget(stats_group)
        main_layout.addLayout(controls_layout)

        # Memory view section
        memory_group = QGroupBox("Memory Visualization")
        memory_layout = QVBoxLayout(memory_group)

        # First-Fit memory view
        self.ff_container = QWidget()
        ff_layout = QVBoxLayout(self.ff_container)
        ff_layout.addWidget(QLabel("First-Fit Memory:"))
        self.ff_view = MemoryView(self.first_fit_memory)
        ff_layout.addWidget(self.ff_view)


        memory_layout.addWidget(self.ff_container)

        # Worst-Fit memory view
        self.wf_container = QWidget()
        wf_layout = QVBoxLayout(self.wf_container)
        wf_layout.addWidget(QLabel("Worst-Fit Memory:"))
        self.wf_view = MemoryView(self.worst_fit_memory)
        wf_layout.addWidget(self.wf_view)
        memory_layout.addWidget(self.wf_container)

        main_layout.addWidget(memory_group)

        # Metrics visualization
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        # Graph selection buttons
        graph_buttons = QHBoxLayout()

        self.util_button = QPushButton("Show Utilization")
        self.util_button.clicked.connect(lambda: self.metrics_graph.plot_utilization())
        graph_buttons.addWidget(self.util_button)

        self.search_button = QPushButton("Show Search Time")
        self.search_button.clicked.connect(lambda: self.metrics_graph.plot_search_time())
        graph_buttons.addWidget(self.search_button)

        metrics_layout.addLayout(graph_buttons)

        # Metrics graph
        self.metrics_graph = MetricsGraph()
        metrics_layout.addWidget(self.metrics_graph)

        main_layout.addWidget(metrics_group)

        self.setCentralWidget(central_widget)

    def strategy_changed(self, text):
        """Handle change of active strategy."""
        if text == "Both":
            self.active_strategy = "both"
            self.ff_container.setVisible(True)
            self.wf_container.setVisible(True)
        elif text == "First-Fit Only":
            self.active_strategy = "first-fit"
            self.ff_container.setVisible(True)
            self.wf_container.setVisible(False)
        elif text == "Worst-Fit Only":
            self.active_strategy = "worst-fit"
            self.ff_container.setVisible(False)
            self.wf_container.setVisible(True)

    def initialize_simulation(self):
        """Initialize simulation with current parameters."""
        # Get values from UI
        self.memory_size = self.memory_size_spin.value()
        self.request_mean = self.req_mean_spin.value()
        self.request_std = self.req_std_spin.value()
        self.sim_speed = self.speed_spin.value()

        # Reset simulation state
        self.reset_simulation()

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        # Stop auto-run if active
        if self.timer.isActive():
            self.toggle_auto_run()

        # Reset state
        self.step_count = 0
        self.first_fit_memory = initialize_memory(self.memory_size, self.request_mean, self.request_std)
        self.worst_fit_memory = self.first_fit_memory.copy()
        self.first_fit_stats = {'allocated': 0, 'failed': 0, 'holes_examined': 0}
        self.worst_fit_stats = {'allocated': 0, 'failed': 0, 'holes_examined': 0}
        self.last_allocation_index_ff = -1
        self.last_release_index_ff = -1
        self.last_allocation_index_wf = -1
        self.last_release_index_wf = -1

        # Update UI
        self.ff_view.set_memory(self.first_fit_memory)
        self.wf_view.set_memory(self.worst_fit_memory)
        self.step_label.setText(str(self.step_count))
        self.update_stats_display()

        # Clear metrics history
        self.metrics_graph.clear_history()

    def step_simulation(self):
        """Execute one step of the simulation."""
        # Generate a memory request
        request_size = max(1, min(self.memory_size - 1,
                                int(random.gauss(self.request_mean, self.request_std))))

        # Update the "current request" label
        self.current_request_label.setText(f"{request_size}")

        # Reset indicators
        self.last_allocation_index_ff = -1
        self.last_release_index_ff = -1
        self.last_allocation_index_wf = -1
        self.last_release_index_wf = -1

        # Apply First-Fit if enabled
        if self.active_strategy in ["both", "first-fit"]:
            ff_success, ff_holes, ff_new_memory, ff_alloc_idx = first_fit(self.first_fit_memory, request_size)
            if ff_success:
                self.first_fit_memory = ff_new_memory
                self.first_fit_stats['allocated'] += 1
                self.first_fit_stats['holes_examined'] += ff_holes
                self.last_allocation_index_ff = ff_alloc_idx
            else:
                self.first_fit_stats['failed'] += 1
                # Release random block on failure
                self.first_fit_memory, release_idx = release_random(self.first_fit_memory)
                self.last_release_index_ff = release_idx

        # Apply Worst-Fit if enabled
        if self.active_strategy in ["both", "worst-fit"]:
            wf_success, wf_holes, wf_new_memory, wf_alloc_idx = worst_fit(self.worst_fit_memory, request_size)
            if wf_success:
                self.worst_fit_memory = wf_new_memory
                self.worst_fit_stats['allocated'] += 1
                self.worst_fit_stats['holes_examined'] += wf_holes
                self.last_allocation_index_wf = wf_alloc_idx
            else:
                self.worst_fit_stats['failed'] += 1
                # Release random block on failure
                self.worst_fit_memory, release_idx = release_random(self.worst_fit_memory)
                self.last_release_index_wf = release_idx

        # Update step count
        self.step_count += 1

        # Update UI with indicators
        self.ff_view.set_memory(self.first_fit_memory, self.last_allocation_index_ff, self.last_release_index_ff)
        self.wf_view.set_memory(self.worst_fit_memory, self.last_allocation_index_wf, self.last_release_index_wf)
        self.step_label.setText(str(self.step_count))
        self.update_stats_display()

        # Calculate metrics for graphs
        ff_util = memory_utilization(self.first_fit_memory)
        wf_util = memory_utilization(self.worst_fit_memory)

        ff_avg_search = 0
        if self.first_fit_stats['allocated'] > 0:
            ff_avg_search = self.first_fit_stats['holes_examined'] / self.first_fit_stats['allocated']

        wf_avg_search = 0
        if self.worst_fit_stats['allocated'] > 0:
            wf_avg_search = self.worst_fit_stats['holes_examined'] / self.worst_fit_stats['allocated']

        # Update metrics graph
        self.metrics_graph.update_metrics(self.step_count, ff_util, wf_util,
                                         ff_avg_search, wf_avg_search)

    def toggle_auto_run(self):
        """Start or stop automatic simulation stepping."""
        if self.timer.isActive():
            self.timer.stop()
            self.run_button.setText("Start Auto")
        else:
            self.timer.start(self.sim_speed)  # Start timer with current speed
            self.run_button.setText("Stop Auto")

    def update_stats_display(self):
        """Update statistics display labels."""
        # First-Fit stats
        ff_allocated = self.first_fit_stats['allocated']
        ff_failed = self.first_fit_stats['failed']
        ff_avg_holes = 0
        if ff_allocated > 0:
            ff_avg_holes = self.first_fit_stats['holes_examined'] / ff_allocated

        self.ff_stats_label.setText(
            f"Allocated: {ff_allocated}, Failed: {ff_failed}, Avg Holes: {ff_avg_holes:.2f}")

        # Worst-Fit stats
        wf_allocated = self.worst_fit_stats['allocated']
        wf_failed = self.worst_fit_stats['failed']
        wf_avg_holes = 0
        if wf_allocated > 0:
            wf_avg_holes = self.worst_fit_stats['holes_examined'] / wf_allocated

        self.wf_stats_label.setText(
            f"Allocated: {wf_allocated}, Failed: {wf_failed}, Avg Holes: {wf_avg_holes:.2f}")

# ----------------------
# main execution
# ----------------------

def main():
    app = QApplication(sys.argv)
    window = MemorySimulator()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


    

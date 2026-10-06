# For visualization
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib import pyplot as plt
import torch

def contour_plot(model, ran = 1):
    if isinstance(ran, int) or isinstance(ran, float):
        X = np.linspace(-ran, ran, 200)
        Y = np.linspace(-ran, ran, 200)
    else:
        X = np.linspace(ran[0], ran[1], 200)
        Y = np.linspace(ran[0], ran[1], 200)
    XX, YY = np.meshgrid(X, Y)
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    with torch.no_grad():
        inputs = torch.FloatTensor(grid_points)
        outputs = model(inputs).numpy()
    ZZ = outputs.reshape(XX.shape)
    plt.contourf(XX, YY, ZZ, levels=50, cmap="bwr", alpha=0.3)
    plt.colorbar(label='Classification')
    plt.title("Decision Boundary Contour Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

class ContourPlotAnim:
    def __init__(self, model, ran=1, N_frames=200):
        self.model = model
        self.ran = ran
        if isinstance(ran, int) or isinstance(ran, float):
            X = np.linspace(-ran, ran, N_frames)
            Y = np.linspace(-ran, ran, N_frames)
        else:
            X = np.linspace(ran[0], ran[1], N_frames)
            Y = np.linspace(ran[0], ran[1], N_frames)
        
        self.XX, self.YY = np.meshgrid(X, Y)
        self.grid_points = np.c_[self.XX.ravel(), self.YY.ravel()]
        
        self.fig, self.ax = plt.subplots()
        plt.ion()
        
        ZZ_init = np.zeros_like(self.XX)
        self.mesh = self.ax.pcolormesh(self.XX, self.YY, ZZ_init, 
                                      cmap="bwr", alpha=0.3, shading='auto')
        self.cbar = self.fig.colorbar(self.mesh, ax=self.ax, label='Classification')

    def update_plot(self):
        with torch.no_grad():
            inputs = torch.FloatTensor(self.grid_points)
            outputs = self.model(inputs).numpy()
        ZZ = outputs.reshape(self.XX.shape)
        
        self.mesh.set_array(ZZ.ravel())
        
        self.mesh.set_clim(vmin=ZZ.min(), vmax=ZZ.max())
        
        self.ax.set_title("Decision Boundary Contour Plot")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.axis('equal')
        self.ax.grid(True)
        plt.draw()
        plt.pause(0.1)

class PlaquePlotAnim:
    def __init__(self, model, grids=16):
        self.model = model
        self.grids = grids
        
        x_edges = np.arange(grids + 1)
        y_edges = np.arange(grids + 1)
        
        x_centers = np.arange(0.5, grids, 1.0) 
        y_centers = np.arange(0.5, grids, 1.0)
        
        self.XX, self.YY = np.meshgrid(x_centers, y_centers)
        self.grid_points = np.c_[self.XX.ravel(), self.YY.ravel()]
        self.grid_points_onehot = np.zeros((grids * grids, grids * 2))
        for i, (x, y) in enumerate(self.grid_points):
            self.grid_points_onehot[i, int(x)] = 1
            self.grid_points_onehot[i, int(y) + grids] = 1
        
        self.fig, self.ax = plt.subplots()
        plt.ion()
        
        colors = ['lightblue', 'lightcoral']
        self.cmap = ListedColormap(colors)
        
        ZZ_init = np.zeros((grids, grids), dtype=int)
        
        self.mesh = self.ax.pcolormesh(
            x_edges,
            y_edges,
            ZZ_init, 
            # cmap=self.cmap, 
            cmap="bwr", 
            shading='flat', 
            edgecolors='black', 
            linewidth=0.5
        )
        self.cbar = self.fig.colorbar(self.mesh, ax=self.ax, label='Classification')
        
        self.ax.set_xlim(0, grids)
        self.ax.set_ylim(0, grids)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Dynamic Two-Color Grid Classification")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        
        plt.show(block=False)

    def update_plot(self):
        with torch.no_grad():
            inputs = torch.FloatTensor(self.grid_points_onehot)
            outputs = self.model(inputs).numpy()
        
        ZZ = outputs.reshape(self.grids, self.grids)
        # ZZ_discrete = np.sign(outputs).astype(int).reshape(self.grids, self.grids)
        
        self.mesh.set_array(ZZ.ravel())
        self.mesh.set_clim(vmin=ZZ.min(), vmax=ZZ.max())
        
        plt.draw()
        plt.pause(0.1)

class MatPlotAnim:
    def __init__(self, size=64):
        """
        简化的热图动画类
        
        Parameters:
        -----------
        size : int
            矩阵尺寸（size x size）
        """
        self.size = size
        
        self.matrix = np.random.rand(size, size)
        
        self.fig, self.ax = plt.subplots()
        plt.ion()
        
        self.im = self.ax.imshow(
            self.matrix,
            cmap='bwr',
            interpolation='none', 
            aspect='auto'
        )
        
        plt.colorbar(self.im, ax=self.ax)
        
        self.ax.set_title("Heatmap Animation")
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
    
    def update(self, new_matrix=None):
        if new_matrix is None:
            self.matrix = np.random.rand(self.size, self.size)
        else:
            if new_matrix.shape != (self.size, self.size):
                raise ValueError(f"Correct shape ({self.size}, {self.size})")
            self.matrix = new_matrix
        
        self.im.set_data(self.matrix)
        
        lim = max(abs(self.matrix.min()), abs(self.matrix.max()))
        self.im.set_clim(vmin=-lim, vmax=lim)
        
        plt.draw()
        plt.pause(0.1)

class DynamicMultiPlot:
    def __init__(self, layout=(1, 1), figsize=(10, 8)):
        self.layout = layout
        self.total_subplots = layout[0] * layout[1]
        
        self.plots = {}
        self.plot_counter = 0
        
        self.fig, self.axes = plt.subplots(
            layout[0], layout[1], 
            figsize=figsize,
            squeeze=False
        )
        self.axes = self.axes.reshape(layout)
        
        plt.ion()
        plt.show()
        
        for i in range(layout[0]):
            for j in range(layout[1]):
                self.axes[i, j].axis('off')
    
    def add_heatmap(self, matrix_size=(16, 16), title="Heatmap", 
                   cmap='viridis', position=None):
        plot_id = self._get_plot_id('heatmap')
        ax = self._get_ax(position)
   
        matrix = np.zeros(matrix_size)
        
        im = ax.imshow(matrix, cmap=cmap, interpolation='nearest', aspect='auto')
        
        cbar = self.fig.colorbar(im, ax=ax)
        
        ax.set_title(title)
        ax.grid(False)
        ax.axis('on')
        if matrix_size[0] == matrix_size[1]:
            ax.set_aspect('equal')
        
        self.plots[plot_id] = {
            'type': 'heatmap',
            'ax': ax,
            'im': im,
            'cbar': cbar,
            'matrix_size': matrix_size,
            'title': title
        }
        
        return plot_id
    
    def add_curve(self, title="Curve", labels=None, colors=None, position=None):
        plot_id = self._get_plot_id('curve')
        ax = self._get_ax(position)
        
        if labels is None:
            labels = ['Curve 1', 'Curve 2']
        if colors is None:
            colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(labels)]
        
        data = {label: [] for label in labels}
        lines = []
        
        for label, color in zip(labels, colors):
            line, = ax.plot([], [], color=color, label=label, linewidth=2)
            lines.append(line)
        
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('on')
        
        self.plots[plot_id] = {
            'type': 'curve',
            'ax': ax,
            'lines': lines,
            'data': data,
            'labels': labels,
            'colors': colors,
            'title': title
        }
        
        return plot_id
    
    def add_grid_plot(self, grids=16, title="Grid Plot", colors=None, cmap=None, position=None):
        plot_id = self._get_plot_id('grid')
        ax = self._get_ax(position)
        
        if colors is None:
            colors = ['lightblue', 'lightcoral']
        
        x = np.linspace(0, grids, grids)
        y = np.linspace(0, grids, grids)
        XX, YY = np.meshgrid(x, y)
        ZZ = np.zeros((grids, grids), dtype=int)
        
        from matplotlib.colors import ListedColormap
        if cmap is None:
            cmap = ListedColormap(colors)
        mesh = ax.pcolormesh(XX, YY, ZZ, cmap=cmap, shading='auto')
        
        ax.set_title(title)
        ax.set_xlim(0, grids)
        ax.set_ylim(0, grids)
        ax.set_aspect('equal')
        ax.axis('on')
        
        self.plots[plot_id] = {
            'type': 'grid',
            'ax': ax,
            'mesh': mesh,
            'grids': grids,
            'XX': XX,
            'YY': YY,
            'title': title
        }
        
        return plot_id
    
    def add_scatter(self, title="Scatter Plot", position=None):
        plot_id = self._get_plot_id('scatter')
        ax = self._get_ax(position)
        
        scatter = ax.scatter([], [], alpha=0.6, edgecolors='w', s=50)
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis('on')
        
        self.plots[plot_id] = {
            'type': 'scatter',
            'ax': ax,
            'scatter': scatter,
            'data': [],
            'title': title
        }
        
        return plot_id
    
    def add_bar(self, labels=None, title="Bar Plot", position=None):
        plot_id = self._get_plot_id('bar')
        ax = self._get_ax(position)
        
        if labels is None:
            labels = ['A', 'B', 'C', 'D']
        
        values = np.zeros(len(labels))
        bars = ax.bar(labels, values, alpha=0.7)
        
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axis('on')
        
        self.plots[plot_id] = {
            'type': 'bar',
            'ax': ax,
            'bars': bars,
            'labels': labels,
            'values': values,
            'title': title
        }
        
        return plot_id
    
    def _get_plot_id(self, plot_type):
        plot_id = f"{plot_type}_{self.plot_counter}"
        self.plot_counter += 1
        return plot_id
    
    def _get_ax(self, position):
        if position is None:
            for i in range(self.layout[0]):
                for j in range(self.layout[1]):
                    if self.axes[i, j] not in [plot['ax'] for plot in self.plots.values()]:
                        return self.axes[i, j]
            raise ValueError("All occupied!")
        
        elif isinstance(position, int):
            i = position // self.layout[1]
            j = position % self.layout[1]
            return self.axes[i, j]
        
        elif isinstance(position, tuple) and len(position) == 2:
            i, j = position
            return self.axes[i, j]
        
        else:
            raise ValueError("Must be int or tuples")
    
    def update_heatmap(self, plot_id, matrix, zero_mean=False, set_lim=None):
        if plot_id not in self.plots or self.plots[plot_id]['type'] != 'heatmap':
            raise ValueError(f"Plot {plot_id} does not exists or is wrong type")
        
        plot_info = self.plots[plot_id]
        plot_info['im'].set_data(matrix)
        
        if set_lim is not None:
            plot_info['im'].set_clim(vmin=set_lim[0], vmax=set_lim[1])
        elif zero_mean:
            max_abs = max(abs(matrix.min()), abs(matrix.max()))
            plot_info['im'].set_clim(vmin=-max_abs, vmax=max_abs)
        else:
            plot_info['im'].set_clim(vmin=matrix.min(), vmax=matrix.max())
    
    def update_curve(self, plot_id, new_data):
        if plot_id not in self.plots or self.plots[plot_id]['type'] != 'curve':
            raise ValueError(f"Plot {plot_id} is not curve type")
        
        plot_info = self.plots[plot_id]
        
        for label, value in new_data.items():
            if label in plot_info['data']:
                plot_info['data'][label].append(value)
        
        for i, label in enumerate(plot_info['labels']):
            if plot_info['data'][label]:
                x_data = range(len(plot_info['data'][label]))
                y_data = plot_info['data'][label]
                plot_info['lines'][i].set_data(x_data, y_data)
        
        plot_info['ax'].relim()
        plot_info['ax'].autoscale_view()
    
    def update_grid(self, plot_id, new_data):
        if plot_id not in self.plots or self.plots[plot_id]['type'] != 'grid':
            raise ValueError(f"Plot {plot_id} does not exists or is not grid type")
        
        plot_info = self.plots[plot_id]
        plot_info['mesh'].set_array(new_data.ravel())
    
    def update_scatter(self, plot_id, new_points):
        if plot_id not in self.plots or self.plots[plot_id]['type'] != 'scatter':
            raise ValueError(f"Plot {plot_id} does not exists or is not scatter type")
        
        plot_info = self.plots[plot_id]
        plot_info['data'].extend(new_points)
        
        if plot_info['data']:
            points_array = np.array(plot_info['data'])
            plot_info['scatter'].set_offsets(points_array)
            plot_info['ax'].relim()
            plot_info['ax'].autoscale_view()
    
    def update_bar(self, plot_id, new_values):
        if plot_id not in self.plots or self.plots[plot_id]['type'] != 'bar':
            raise ValueError(f"Plot {plot_id} does not exists or is not bar type")
        
        plot_info = self.plots[plot_id]
        plot_info['values'] = new_values
        
        for bar, value in zip(plot_info['bars'], new_values):
            bar.set_height(value)
        
        max_value = max(new_values) if len(new_values) > 0 else 1
        plot_info['ax'].set_ylim(0, max_value * 1.1)
    
    def update_from_model(self, plot_id, model, input_data=None):
        plot_info = self.plots[plot_id]
        
        if plot_info['type'] == 'grid':
            grids = plot_info['grids']
            XX, YY = plot_info['XX'], plot_info['YY']
            grid_points = np.c_[XX.ravel(), YY.ravel()]
            
            with torch.no_grad():
                inputs = torch.FloatTensor(grid_points)
                outputs = model(inputs).numpy()
            
            new_data = (outputs > 0.5).astype(int).reshape((grids, grids))
            self.update_grid(plot_id, new_data)
        
        elif plot_info['type'] == 'heatmap':
            matrix_size = plot_info['matrix_size']
            
            if input_data is None:
                x = np.linspace(-1, 1, matrix_size[1])
                y = np.linspace(-1, 1, matrix_size[0])
                xx, yy = np.meshgrid(x, y)
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                input_data = grid_points
            
            with torch.no_grad():
                inputs = torch.FloatTensor(input_data)
                outputs = model(inputs).numpy()
            
            new_matrix = outputs.reshape(matrix_size)
            self.update_heatmap(plot_id, new_matrix)
    
    def refresh(self):
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def get_plot_info(self, plot_id=None):
        if plot_id is None:
            return self.plots
        else:
            return self.plots.get(plot_id)
    
    def close(self):
        plt.ioff()
        plt.show()
        plt.close(self.fig)

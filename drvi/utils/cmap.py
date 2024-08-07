from matplotlib.colors import LinearSegmentedColormap

cmap_data = {
    'red':   ((0.0, 0.0, 0.0),
              (0.25, 0.0, 0.0),
              (0.5, 1.0, 1.0),
              (0.75, 1.0, 1.0),
              (1.0, 0.5, 0.0)),

    'green': ((0.0, 0.0, 0.0),
              (0.25, 0.0, 0.0),
              (0.5, 1.0, 1.0),
              (0.75, 0.0, 0.0),
              (1.0, 0.0, 0.0)),

    'blue':  ((0.0, 0.0, 0.5),
              (0.25, 1.0, 1.0),
              (0.5, 1.0, 1.0),
              (0.75, 0.0, 0.0),
              (1.0, 0.0, 0.0))
}

saturated_red_blue_cmap = LinearSegmentedColormap('SaturatedRdBu', cmap_data)

import plotly.graph_objects as go
import numpy as np
import torch

def data_generation_animation_2d(generated_data_list, real_data_sample_for_plot, use_case = "gan"):
    """
    Creates an animation showing the evolution of generated 2D data points.

    Args:
        generated_data_list (list): A list of numpy arrays, each containing generated 2D data for a snapshot.
        real_data_sample_for_plot (np.ndarray): A numpy array containing real 2D data points for comparison.
    """
    if use_case == "gan":
      dd = {"gen": "Generated Data","real": "Real Data","title": "GAN Training: Generated vs. Real 2D Data Over Time (Concentric Circles)"}
    elif use_case == "forward_diffusion":
      dd = {"gen": "Current Data State","real": "Target Dataset","title": "Evolution of Data in Forward Diffusion Process"}
    elif use_case == "backward_diffusion":
      dd = {"gen": "Current Data State","real": "Target Dataset","title": "Evolution of Data in Backward Diffusion Process"}
    
    frames = [
        go.Frame(
            data=[
                go.Scatter(x=generated_data_plot[:, 0], y=generated_data_plot[:, 1], mode='markers', name=dd["gen"]),
                go.Scatter(x=real_data_sample_for_plot[:, 0], y=real_data_sample_for_plot[:, 1], mode='markers', name=dd["real"])#, marker=dict(size=5, opacity=0.7)
            ],
            name=f"Epoch {epoch}"
        ) for generated_data_plot, epoch in zip(generated_data_list, range(len(generated_data_list)))
    ]

    if frames:
        print("Generating 2D animation...")
        fig = go.Figure(
            data=[
                go.Scatter(x=generated_data_list[0][:, 0] if generated_data_list else [], y=generated_data_list[0][:, 1] if generated_data_list else [], mode='markers', name=dd["gen"], marker=dict(size=5, opacity=0.7)),
                go.Scatter(x=real_data_sample_for_plot[:, 0], y=real_data_sample_for_plot[:, 1], mode='markers', name=dd["real"], marker=dict(size=5, opacity=0.7))
            ],
            layout=go.Layout(
                title_text=dd["title"],
                xaxis=dict(title="X-axis"),
                yaxis=dict(title="Y-axis", scaleanchor="x", scaleratio=1), # Keep aspect ratio 1:1
                 updatemenus=[{
                    "type": "buttons",
                    "buttons": [{
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                    }, {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                    }]
                }]
            ),
            frames=frames
        )
        t_arr_np = np.array(range(len(generated_data_list)))/(len(generated_data_list)-1)
        # Add slider
        fig.update_layout(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Step: " if use_case == "gan" else "Time: "},
                steps=[
                    dict(
                        method="animate",
                        args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 0}}],
                        label=f"{t_arr_np[i]:.2f}"
                    ) for i, f in enumerate(frames)
                ]
            )]
        )
        fig.show()
    else:
        print("No frames to create the 2D animation.")


# --- Final 2D Scatter Plot ---
def final_scatter_plot_2d(generated_data_for_plot, real_data_sample_for_plot, use_case = "gan"):
    """
    Creates a final scatter plot comparing the last generated 2D data with real data.

    Args:
        generated_data_for_plot (np.ndarray): The last generated 2D data.
        real_data_sample_for_plot (np.ndarray): Real 2D data for comparison.
    """
    if use_case == "gan":
        dd = {"gen": "Generated Data","real": "Real Data","title": "Final Generated vs. Real 2D Data (Concentric Circles)"}
    else:
      dd = {"gen": "Reference Dataset","real": "Target Dataset","title": "Reference vs. Target Dataset (Concentric Circles)"}
    fig = go.Figure()

    fig.add_trace(go.Scattergl(x=generated_data_for_plot[:, 0], y=generated_data_for_plot[:, 1], mode='markers', name=dd["gen"], marker=dict(size=5, opacity=0.7)))
    fig.add_trace(go.Scattergl(x=real_data_sample_for_plot[:, 0], y=real_data_sample_for_plot[:, 1], mode='markers', name=dd["real"], marker=dict(size=5, opacity=0.7)))

    fig.update_layout(
        title=dd["title"],
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        yaxis=dict(scaleanchor="x", scaleratio=1), # Keep aspect ratio 1:1
    )
    fig.show()


def plot_discriminator_decision_boundary(discriminator, real_data_sample, generated_data_sample, device, resolution=100):
    """
    Plots the decision boundary of the discriminator in 2D space.

    Args:
        discriminator (torch.nn.Module): The trained discriminator.
        real_data_sample (np.ndarray): Sample of real data (N, 2).
        generated_data_sample (np.ndarray): Sample of generated data (N, 2).
        resolution (int): The number of points to sample along each axis.
    """
    print("Plotting Discriminator Decision Boundary...")
    # Determine the range for the grid
    all_data = np.vstack([real_data_sample, generated_data_sample])
    x_min, x_max = all_data[:, 0].min() - 0.5, all_data[:, 0].max() + 0.5
    y_min, y_max = all_data[:, 1].min() - 0.5, all_data[:, 1].max() + 0.5

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                           np.linspace(y_min, y_max, resolution))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    # Get discriminator output for each point on the grid
    discriminator.eval()
    with torch.no_grad():
        Z = discriminator(grid_points).cpu().numpy().reshape(xx.shape)
    discriminator.train()

    # Create the plot
    fig = go.Figure()

    # Add contour plot of the discriminator output
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z,
        colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
        colorbar=dict(title='Discriminator Output (0=Fake, 1=Real)'),
        contours=dict(
            coloring='heatmap',
            showlabels=True, # show labels on contours
            labelfont = dict( # label font properties
                size = 12,
                color = 'white',
            )
        ),
        opacity=0.6
    ))

    # Add real and generated data points
    fig.add_trace(go.Scattergl(
        x=real_data_sample[:, 0],
        y=real_data_sample[:, 1],
        mode='markers',
        name='Real Data',
        marker=dict(
            size=5,
            opacity=0.8,
            color='red'
        )
    ))
    fig.add_trace(go.Scattergl(
        x=generated_data_sample[:, 0],
        y=generated_data_sample[:, 1],
        mode='markers',
        name='Generated Data',
        marker=dict(
            size=5,
            opacity=0.8,
            color='blue'
        )
    ))


    fig.update_layout(
        title="Discriminator Decision Boundary",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        yaxis=dict(scaleanchor="x", scaleratio=1), # Keep aspect ratio 1:1
        hovermode='closest'
    )
    fig.show()
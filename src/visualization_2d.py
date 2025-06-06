import plotly.graph_objects as go
import numpy as np
import torch
import plotly.figure_factory as ff 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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
                go.Scatter(x=real_data_sample_for_plot[:, 0], y=real_data_sample_for_plot[:, 1], mode='markers', name=dd["real"], marker=dict(size=5, opacity=0.7, color="red")),
                go.Scatter(x=generated_data_plot[:, 0], y=generated_data_plot[:, 1], mode='markers', name=dd["gen"], marker=dict(size=5, opacity=0.7, color="blue"))
            ],
            name=f"Epoch {epoch}"
        ) for generated_data_plot, epoch in zip(generated_data_list, range(len(generated_data_list)))
    ]

    if frames:
        print("Generating 2D animation...")
        fig = go.Figure(
            data=[
                go.Scatter(x=real_data_sample_for_plot[:, 0], y=real_data_sample_for_plot[:, 1], mode='markers', name=dd["real"], marker=dict(size=5, opacity=0.7, color="red")),
                go.Scatter(x=generated_data_list[0][:, 0] if generated_data_list else [], y=generated_data_list[0][:, 1] if generated_data_list else [], mode='markers', name=dd["gen"], marker=dict(size=5, opacity=0.7, color="blue"))
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

    fig.add_trace(go.Scattergl(x=real_data_sample_for_plot[:, 0], y=real_data_sample_for_plot[:, 1], mode='markers', name=dd["real"], marker=dict(size=5, opacity=0.7, color="red")))
    fig.add_trace(go.Scattergl(x=generated_data_for_plot[:, 0], y=generated_data_for_plot[:, 1], mode='markers', name=dd["gen"], marker=dict(size=5, opacity=0.7, color="blue")))

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
        device (torch.device): The device (e.g., 'cpu' or 'cuda') to run the discriminator on.
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
        colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']], # Blue for fake, Red for real
        colorbar=dict(title='Discriminator Output (0=Fake, 1=Real)'),
        contours=dict(
            coloring='heatmap',
            showlabels=True, # show labels on contours
            labelfont = dict( # label font properties
                size = 12,
                color = 'white',
            )
        ),
        opacity=0.6,
        name='Discriminator Output' # Name for the contour trace in the legend
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
        hovermode='closest',
        
        legend=dict(
            x=0, # Position legend at the left edge
            y=1, # Position legend at the top edge
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.7)', # Optional: add a semi-transparent background
            bordercolor='Black',
            borderwidth=1
        )
    )
    fig.show()


def plot_quiver_initial_step(ema_model, device):
  xmin, xmax = -4, 4
  ymin, ymax = -4, 4
  grid_resolution = 20 # Number of points along each axis (for 20j)
  t0_val = 0.05

  # Create the grid
  yy, xx = np.mgrid[ymin:ymax:complex(0, grid_resolution), xmin:xmax:complex(0, grid_resolution)]
  x_flat = np.vstack([xx.ravel(), yy.ravel()]).T
  x_tensor = torch.from_numpy(x_flat).float().to(device)
  t_field = torch.full((x_tensor.shape[0],), t0_val, device=device)

  with torch.no_grad():
      score_field = ema_model(t_field, x_tensor).cpu().numpy()

  u_vec_raw = score_field[:, 0]
  v_vec_raw = score_field[:, 1]

  # --- Scaling for ff.create_quiver ---
  magnitudes = np.sqrt(u_vec_raw**2 + v_vec_raw**2)
  max_raw_magnitude = np.max(magnitudes)

  if max_raw_magnitude == 0:
      plotly_quiver_scale_factor = 1.0 # Default, u/v are zero anyway
  else:
      # Desired length of the longest arrow in data units
      plot_width_data_units = xmax - xmin
      desired_max_arrow_length_on_plot = plot_width_data_units / (grid_resolution * 1.5)
      plotly_quiver_scale_factor = desired_max_arrow_length_on_plot / max_raw_magnitude

  fig_quiver = ff.create_quiver(
      x=xx.ravel(),
      y=yy.ravel(),
      u=u_vec_raw, # Pass original vectors
      v=v_vec_raw, # Pass original vectors
      scale=plotly_quiver_scale_factor, # Calculated scale factor
      arrow_scale=0.3,          # Size of arrowhead (fraction of stem length)
      name='Score Field',
      line_width=1,
  )

  strtitle_quiver = f"Score field at time t={t0_val} (Plotly)"
  fig_quiver.update_layout(
      title=strtitle_quiver,
      xaxis=dict(range=[xmin, xmax], showgrid=False, zeroline=False),
      yaxis=dict(
          range=[ymin, ymax],
          scaleanchor="x",  # For 'equal' aspect
          scaleratio=1,     # For 'equal' aspect
          showgrid=False,
          zeroline=False
      ),
  )
  fig_quiver.show()

def show_image_grid(images_tensor, nrow=8, title=""):
    """Imshow for Tensor. Images are expected to be in [-1, 1] range."""
    grid_img = make_grid(images_tensor.cpu(), nrow=nrow, normalize=True, value_range=(-1,1), padding=2)
    plt.figure(figsize=(nrow, images_tensor.size(0)//nrow + 1))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()
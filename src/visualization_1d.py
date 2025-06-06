
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch

def data_generation_animation(generated_data_list, real_data_sample_for_plot, binsize=0.1):
  # --- Create and Display Animation (if frames were captured) ---
  frames = [
      go.Frame(
          data=[
                  go.Histogram(x=generated_data_plot[:,0], xbins={"size":binsize}, opacity=0.7, histnorm='probability density', name=f"Datos Generados"),
                  go.Histogram(x=real_data_sample_for_plot, xbins={"size":binsize}, opacity=0.7, histnorm='probability density', name="Datos Reales")
              ],
              name=f"Epoch {epoch}"
          ) for generated_data_plot, epoch in zip(generated_data_list, range(len(generated_data_list)))
  ]

  if frames:
      print("Generando la animación...")
      fig = go.Figure(
          data=[
              go.Histogram(x=generated_data_list[0][:,0] if generated_data_list else [], xbins={"size":binsize}, opacity=0.7, histnorm='probability density', name="Generated (Initial)"),
              go.Histogram(x=real_data_sample_for_plot, xbins={"size":binsize}, opacity=0.7, histnorm='probability density', name="Real Data")
          ],
          layout=go.Layout(
              title_text="Entrenamiento de GANs: Distribución Generada vs Data Real a lo largo del entrenamiento",
              xaxis=dict(title="Valor", range=[real_data_sample_for_plot.min()-1, real_data_sample_for_plot.max()+1]),
              yaxis=dict(title="Frecuencia", range=[0,1]),
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
      fig.update_layout(
          barmode='overlay', # Overlay histograms
      )
      t_arr_np = np.array(range(len(generated_data_list)))/(len(generated_data_list)-1)
      # Add slider
      fig.update_layout(
          sliders=[dict(
              active=0,
              currentvalue={"prefix": "Step: "},
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
      print("NO hay frames para hacer la animación")

# Visualización de los datos generados
def final_histogram_with_density(generated_data_for_plot, real_data_sample_for_plot, data_distribution, mapeo_distribuciones, binsize=0.1):
  fig = go.Figure()
  fig.add_trace(go.Histogram(x=generated_data_for_plot, xbins={"size":binsize}, histnorm='probability density', name='Datos generados'))
  fig.add_trace(go.Histogram(x=real_data_sample_for_plot, xbins={"size":binsize}, histnorm='probability density', name='Datos reales'))

  x_pdf = np.linspace(min(real_data_sample_for_plot.min(), generated_data_for_plot.min()) -1,max(real_data_sample_for_plot.max(), generated_data_for_plot.max()) +1, 500)

  scipy_name, dist, params = mapeo_distribuciones[data_distribution]
  fig.add_trace(go.Scatter(x=x_pdf, y=dist.pdf(x_pdf, *params), mode='lines', name=f"Densidad Teórica ({data_distribution.capitalize()})", line=dict(color='red', width=2)))

  fig.update_traces(opacity=0.7)
  fig.update_layout(
      title="Distribución de datos generados vs. datos reales",
      xaxis_title="Valor",
      yaxis_title="Frecuencia")
  fig.show()

def plot_test_evolution(shapiro_statistics, shapiro_p_values):
  epochs_snapshots = np.arange(0, len(shapiro_statistics))

  fig_shapiro = go.Figure()
  fig_shapiro.add_trace(go.Scatter(x=epochs_snapshots, y=shapiro_statistics, mode='lines+markers', name='Estadístico Kolmogorov-Smirnov'))
  fig_shapiro.add_trace(go.Scatter(x=epochs_snapshots, y=shapiro_p_values, mode='lines+markers', name='P-valor Kolmogorov-Smirnov', yaxis='y2'))

  fig_shapiro.update_layout(
      title=f"Evolución del Test de Kolmogorov-Smirnov durante el entrenamiento de la GAN",
      xaxis_title="Snapshot",
      yaxis_title="Estadístico",
      yaxis2=dict(title="P-valor", overlaying='y', side='right', range=[0,1]),
      legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
  )
  fig_shapiro.show()


def plot_mean_and_std_evolution(trajectories):
    mean_sq_dist = np.zeros(len(trajectories))
    std_sq_dist = np.zeros(len(trajectories))

    for k, samples in enumerate(trajectories):
        batch_mean = samples.mean(axis=0)  # Mean across batch for each dimension
        mean_sq_dist[k] = (batch_mean**2).sum()
    
        batch_std = samples.std(axis=0)  # Std across batch for each dimension
        std_sq_dist[k] = ((batch_std - 1.0)**2).sum()
    
    t_arr_np = np.array(range(len(trajectories)))
    fig_plotly = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Squared L2 dist of mean from 0", "Squared L2 dist of std from 1")
    )

    mean_plot_data = np.abs(mean_sq_dist)

    fig_plotly.add_trace(
        go.Scatter(
            x=t_arr_np,
            y=mean_plot_data,
            mode='lines', # We want a line plot
            name='Mean Sq Dist', # Optional: name for legend if shown
        ),
        row=1, col=1 # Specify which subplot this trace belongs to
    )

    std_plot_data = np.abs(std_sq_dist)

    fig_plotly.add_trace(
        go.Scatter(
            x=t_arr_np,
            y=std_plot_data,
            mode='lines', # We want a line plot
            name='Std Sq Dist', # Optional: name for legend if shown
        ),
        row=1, col=2 # Specify which subplot this trace belongs to
    )

    fig_plotly.show()


def plot_gan_losses(d_losses, g_losses, save_path=None):
    """
    Plots the evolution of discriminator and generator losses over training epochs using Plotly.

    Args:
        d_losses (list or array): Discriminator losses per epoch.
        g_losses (list or array): Generator losses per epoch.
        save_path (str, optional): If provided, saves the figure to this path (as HTML).
    """
    epochs = list(range(1, len(d_losses) + 1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=d_losses,
        mode='lines',
        name='Discriminator Loss',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=g_losses,
        mode='lines',
        name='Generator Loss',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='GAN Training Losses Over Time',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )

    if save_path:
        fig.write_html(save_path)
    fig.show()
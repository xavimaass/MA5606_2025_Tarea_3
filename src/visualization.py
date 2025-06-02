
import plotly.graph_objects as go

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
                      "label": "Play",
                      "method": "animate",
                      "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                  }]
              }]
          ),
          frames=frames
      )
      fig.update_layout(
          barmode='overlay', # Overlay histograms
      )
      fig.show()
  else:
      print("NO hay frames para hacer la animación")
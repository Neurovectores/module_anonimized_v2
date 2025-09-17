from pyvis.network import Network
import tempfile
import webbrowser
import os
import time


class PaintNV:
    def __init__(self, notebook=True, height="800px", width="100%"):
        """
        Inicializa el grafo del neurovector.
        """
        self.net = Network(notebook=notebook, height=height, width=width, directed=True,cdn_resources='remote')
        self._configure_layout()

    def _configure_layout(self):
        """
        Configura el layout circular para el grafo.
        """
        self.net.set_options('''
        var options = {
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.005,
              "springLength": 200,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            }
          },
          "layout": {
            "improvedLayout": true
          },
          "nodes": {
            "font": {
              "size": 16,
              "face": "arial",
              "align": "center",
              "multi": true
            }
          }
        }
        ''')

    def draw_neurovector(self, neurovector):
        """
        Método principal para pintar la estructura del neurovector.
        """
        self._draw_neurovector_node(neurovector)
        self._draw_axons(neurovector)
        self._draw_nodes_and_vectors(neurovector)

    def _draw_neurovector_node(self, neurovector):
        """
        Dibuja el nodo principal del Neurovector.
        """
        nv_label = (
            f"Neurovector {neurovector.id}\n"
            f"Acc Use: {neurovector.acc_use}\n"
            f"Acc Successes: {neurovector.acc_successes}\n"
            f"Acc Hits: {neurovector.acc_hits}\n"
            f"Energy: {neurovector.energy:.2f}\n"
            f"RMSE: {neurovector.RMSE:.2f}\n"
            f"MAE: {neurovector.MAE:.2f}\n"
            f"Certainty: {neurovector.certainty:.2f}"
        )

        self.net.add_node(
            f"Neurovector {neurovector.id}",
            label=nv_label,
            color="#ADD8E6",
            shape="ellipse",
        )

    def _draw_axons(self, neurovector):
        """
        Dibuja los axones y las flechas entre el neurovector y los axones.
        """
        def interpolate_color(value, start_color, end_color):
            """
            Interpola entre dos colores en función de un valor entre 0 y 1.
            """
            r1, g1, b1 = start_color
            r2, g2, b2 = end_color
            r = int(r1 + (r2 - r1) * value)
            g = int(g1 + (g2 - g1) * value)
            b = int(b1 + (b2 - b1) * value)
            return f"#{r:02x}{g:02x}{b:02x}"

        for axon_name, axon in neurovector.axons.items():
            axon_label = f"{axon_name}"
            
            # Crear el contenido del tooltip
            axon_tooltip = (
                f"Axon {axon_name}\n"
                f"Acc Use: {axon.acc_use}\n"
                f"Acc Successes: {axon.acc_successes}\n"
                f"Energy: {axon.energy:.2f}\n"
                f"MAE: {axon.MAE:.2f}\n"
                f"RMSE: {axon.RMSE:.2f}\n"
                f"Acc Residue: {axon.acc_residue}\n"
                f"Acc Output: {axon.acc_output}"
            )

            # Calcular el color del axón basado en el valor de energy
            color = interpolate_color(axon.energy, (200, 255, 200), (0, 255, 0))  # Gradiente de verde claro a verde intenso

            # Agregar el nodo del axón con un tooltip
            self.net.add_node(
                axon_label,
                label=axon_label,  # Texto visible: el nombre del axón
                title=axon_tooltip,  # Tooltip con información completa
                color=color,
                shape="ellipse",
            )

            # Determinar la dirección de la flecha entre el neurovector y el axón
            if axon.acc_output > 0:
                # Salida: del neurovector al axón
                self.net.add_edge(
                    f"Neurovector {neurovector.id}",
                    axon_label,
                    label=axon_name,
                    color="#FFD700",  # Dorado para salida
                )
            else:
                # Entrada: del axón al neurovector
                self.net.add_edge(
                    axon_label,
                    f"Neurovector {neurovector.id}",
                    label=axon_name,
                    color="#1f77b4",  # Azul para entrada
                )




    def _draw_nodes_and_vectors(self, neurovector):
        """
        Dibuja los nodos asociados a los axones y las flechas (vectores) entre ellos.
        """
        def interpolate_color(value, start_color, end_color):
            """
            Interpola entre dos colores en función de un valor entre 0 y 1.
            """
            r1, g1, b1 = start_color
            r2, g2, b2 = end_color
            r = int(r1 + (r2 - r1) * value)
            g = int(g1 + (g2 - g1) * value)
            b = int(b1 + (b2 - b1) * value)
            return f"#{r:02x}{g:02x}{b:02x}"

        for axon_name, axon in neurovector.axons.items():
            axon_label = f"{axon_name}"

            for node in axon.nodes:
                node_label = f"Node {node.id}"
                
                # Crear el contenido del tooltip para el nodo
                node_tooltip = (
                    f"Node {node.name}\n"
                    f"Acc Use: {node.acc_use}\n"
                    f"Acc Use Var: {node.acc_use_var}\n"
                    f"Acc Successes: {node.acc_successes}\n"
                    f"Acc Residue: {node.acc_residue}\n"
                    f"Energy: {node.energy:.2f}\n"
                    f"RMSE: {node.RMSE:.2f}\n"
                    f"MAE: {node.MAE:.2f}"
                )

                # Calcular el color del nodo basado en el valor de energy
                color = interpolate_color(node.energy, (255, 200, 200), (255, 0, 0))  # Gradiente de rojo claro a rojo intenso

                # Agregar el nodo con el tooltip
                self.net.add_node(
                    node_label,
                    label=node.name,
                    title=node_tooltip,  # Tooltip con información completa
                    color=color,
                    size=10,
                )

                # Determinar el tipo de flecha entre el nodo y el axón
                if node.id in neurovector.vectors:
                    vector = neurovector.vectors[node.id]
                    if vector.acc_output > 0:
                        # Salida: del nodo al axón
                        self.net.add_edge(
                            axon_label,
                            node_label,
                            label=f"Vector {vector.name}",
                            color="#FFD700",
                        )
                    else:
                        # Entrada: del axón al nodo
                        self.net.add_edge(
                            node_label,
                            axon_label,
                            label=f"Vector {vector.name}",
                            color="#1f77b4",
                        )




    def show(self, delay=5):
        """
        Genera un archivo temporal HTML, lo abre en el navegador y luego lo elimina después de un retraso.
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as tmp_file:
            self.net.write_html(tmp_file.name)  # Escribir contenido HTML en el archivo temporal
            tmp_file_path = tmp_file.name  # Guardar la ruta para eliminarla después

        webbrowser.open(tmp_file_path)  # Abrir el archivo temporal en el navegador

        # Esperar antes de eliminar el archivo
        time.sleep(delay)

        # Eliminar el archivo temporal
        try:
            os.unlink(tmp_file_path)
        except OSError as e:
            print(f"Error al eliminar el archivo temporal: {e}")

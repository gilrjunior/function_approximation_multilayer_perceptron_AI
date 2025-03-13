import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

from Mlp import * # Import da classe Mlp no mesmo pacote mlp

class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Multilayer Perceptron")
        self.root.geometry("1300x900")
        self.root.configure(bg="#F0F0F0")

        # Flag para indicar se está treinando
        self.is_training = False

        # Flag para parar treinamento
        self.stop_training_flag = False

        self.setup_styles()
        self.create_frames()
        self.create_controls()
        self.create_graphs()

        # Rótulo para mostrar a época atual
        self.epoch_label = ttk.Label(self.frame_controls, text="Época: 0", style="TLabel")
        self.epoch_label.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.error_label = ttk.Label(self.frame_controls, text="Erro: 0", style="TLabel")
        self.error_label.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Vetor de erros (só para exibir se quisermos armazenar)
        self.error_history = []

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#F0F0F0")
        style.configure("TLabel", background="#F0F0F0", foreground="#333333", font=("Arial", 12))
        style.configure("Rounded.TEntry",
                        fieldbackground="white",
                        bordercolor="#CCCCCC",
                        lightcolor="#CCCCCC",
                        foreground="#000000",
                        padding=5,
                        borderwidth=2,
                        relief="solid")
        style.layout("Rounded.TEntry", [
            ("Entry.border", {
                "sticky": "nswe",
                "children": [
                    ("Entry.padding", {
                        "sticky": "nswe",
                        "children": [
                            ("Entry.textarea", {"sticky": "nswe"})
                        ]
                    })
                ]
            })
        ])
        style.configure("Red.TButton",
                        foreground="white",
                        background="#FF0000",
                        padding=5,
                        borderwidth=2,
                        relief="solid",
                        anchor="center")
        style.map("Red.TButton",
            foreground=[("active", "black")],
            background=[("active", "white")]
        )
        style.layout("Red.TButton", [
            ("Button.border", {
                "sticky": "nswe",
                "children": [
                    ("Button.padding", {
                        "sticky": "nswe",
                        "children": [
                            ("Button.label", {"sticky": "nswe"})
                        ]
                    })
                ]
            })
        ])

    def create_frames(self):
        self.frame_controls = ttk.Frame(self.root, padding=10, style="TFrame")
        self.frame_controls.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        self.frame_true_graph = ttk.Frame(self.root, padding=10, style="TFrame")
        self.frame_true_graph.grid(row=0, column=1, sticky="ne", padx=10, pady=10)

        self.frame_overlap = ttk.Frame(self.root, padding=10, style="TFrame")
        self.frame_overlap.grid(row=1, column=0, sticky="sw", padx=10, pady=10)

        self.frame_error = ttk.Frame(self.root, padding=10, style="TFrame")
        self.frame_error.grid(row=1, column=1, sticky="se", padx=10, pady=10)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

    def create_controls(self):
        labels_and_defaults = [
            ("Taxa de Aprendizado:", "0.01"),
            ("Erro Máximo Tolerado:", "0.05"),
            ("Quantidade de Neurônios:", "200"),
            ("Tamanho da Amostra:", "50"),
            ("Valor Mínimo de X:", "0"),
            ("Valor Máximo de X:", "3.14")
        ]

        self.entries = {}
        for idx, (text, default_value) in enumerate(labels_and_defaults):
            lbl = ttk.Label(self.frame_controls, text=text, style="TLabel")
            lbl.grid(row=idx, column=0, padx=5, pady=5, sticky="w")

            entry = ttk.Entry(self.frame_controls, style="Rounded.TEntry", width=25, font=("Arial", 12))
            entry.grid(row=idx, column=1, padx=5, pady=5, sticky="w")

            # Insere o valor padrão no campo de texto
            entry.insert(0, default_value)

            # Armazena a referência ao Entry em um dicionário
            self.entries[text] = entry

        btn_approx = ttk.Button(
            self.frame_controls,
            text="Realizar Aproximação",
            style="Red.TButton",
            width=20,
            command=self.start_training
        )
        btn_approx.grid(row=len(labels_and_defaults), column=0, columnspan=2, padx=5, pady=10, sticky="w")

        # Botão para Parar Treinamento
        btn_stop = ttk.Button(
            self.frame_controls,
            text="Parar",
            width=20,
            command=self.stop_training
        )
        btn_stop.grid(row=len(labels_and_defaults), column=1, columnspan=1, padx=5, pady=10, sticky="w")

        

    def create_graphs(self):
        # Gráfico verdadeiro
        self.fig_true, self.ax_true = plt.subplots(figsize=(6, 6))
        self.ax_true.set_title("Gráfico verdadeiro")
        self.ax_true.set_xlabel("X - Entrada")
        self.ax_true.set_ylabel("Y - Saída")
        self.canvas_true = FigureCanvasTkAgg(self.fig_true, master=self.frame_true_graph)
        self.canvas_true.get_tk_widget().pack()

        # Gráfico sobreposto
        self.fig_overlap, self.ax_overlap = plt.subplots(figsize=(6, 6))
        self.ax_overlap.set_title("Gráfico Sobreposto (Verdadeiro vs MLP)")
        self.ax_overlap.set_xlabel("X - Entrada")
        self.ax_overlap.set_ylabel("Y - Saída")
        self.canvas_overlap = FigureCanvasTkAgg(self.fig_overlap, master=self.frame_overlap)
        self.canvas_overlap.get_tk_widget().pack()

        # Gráfico de Erro x Épocas
        self.fig_error, self.ax_error = plt.subplots(figsize=(6, 6))
        self.ax_error.set_title("Erro x Épocas")
        self.ax_error.set_xlabel("Épocas")
        self.ax_error.set_ylabel("Erro")
        self.canvas_error = FigureCanvasTkAgg(self.fig_error, master=self.frame_error)
        self.canvas_error.get_tk_widget().pack()

    def stop_training(self):
        self.stop_training_flag = True
        self.is_training = False

    def on_closing(self):
        """Força o encerramento do aplicativo ao fechar a janela."""
        print("Fechando o aplicativo...")

        self.stop_training_flag = True  # Sinaliza para parar o treinamento

        if hasattr(self, "training_thread") and self.training_thread.is_alive():
            print("Aguardando thread finalizar...")
            self.training_thread.join(timeout=2)  # Espera no máximo 2 segundos para encerrar

        self.root.destroy()  # Fecha a janela do Tkinter
        print("Aplicação encerrada.")

        os._exit(0)  # Mata o processo Python imediatamente

    def should_stop(self):
        return self.stop_training_flag

    def start_training(self):
        # Se já estiver treinando, não faz nada (ou avisa o usuário)
        if self.is_training:
            print("Já existe um treinamento em andamento.")
            return
        
        self.stop_training_flag = False
        self.is_training = True
        
        self.epoch_label.config(text="Época: 0  ")
        self.error_label.config(text="Erro: 0  ")
        
        # Lê os parâmetros
        lr = float(self.entries["Taxa de Aprendizado:"].get())
        min_error = float(self.entries["Erro Máximo Tolerado:"].get())
        num_neurons = int(self.entries["Quantidade de Neurônios:"].get())
        sample_size = int(self.entries["Tamanho da Amostra:"].get())
        x_min = float(self.entries["Valor Mínimo de X:"].get())
        x_max = float(self.entries["Valor Máximo de X:"].get())

        # Instancia a MLP
        self.mlp = Mlp(num_neurons, sample_size, x_min, x_max, lr)

        # Constrói o gráfico de targets
        self.ax_true.clear()
        self.ax_true.set_title("Gráfico verdadeiro")
        self.ax_true.set_xlabel("X - Entrada")
        self.ax_true.set_ylabel("Y - Saída")
        self.ax_true.plot(self.mlp.inputs, self.mlp.targets, 'b-', label='Função Real')
        self.canvas_true.draw()

        # Inicia o treinamento e passa o callback
        # Criar uma thread separada para rodar o treinamento
        training_thread = threading.Thread(
            target=self.mlp.optimized_train,
            args=(min_error, self.update_training_status, self.should_stop),
            daemon=True  # Thread daemon para encerrar automaticamente quando o programa fechar
        )

        training_thread.start()


    def update_training_status(self, epoch, error_history, approx):
        """
        Este método será chamado a cada época pelo Mlp.train()
        """
        # Atualiza o rótulo de época
        self.epoch_label.config(text=f"Época: {epoch}")

        self.error_label.config(text=f"Erro: {error_history[-1]:.4f}")

        # Atualiza o gráfico de erro
        self.ax_error.clear()
        self.ax_error.set_title("Erro x Épocas")
        self.ax_error.set_xlabel("Épocas")
        self.ax_error.set_ylabel("Erro")
        self.ax_error.plot(range(1, epoch + 1), error_history, 'r-')
        self.canvas_error.draw()

        # 3) Atualiza gráfico sobreposto (Verdadeiro vs MLP)
        self.ax_overlap.clear()
        self.ax_overlap.set_title("Gráfico Sobreposto (Verdadeiro vs MLP)")
        self.ax_overlap.set_xlabel("X - Entrada")
        self.ax_overlap.set_ylabel("Y - Saída")

        # Plot da função verdadeira
        self.ax_overlap.plot(self.mlp.inputs, self.mlp.targets, 'b-', label='Função Real')

        # Plot da aproximação da MLP
        self.ax_overlap.plot(self.mlp.inputs, approx, 'r--', label='MLP Approx')
        self.ax_overlap.legend()
        self.canvas_overlap.draw()

        # Faz update da interface para não travar
        self.root.update_idletasks()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
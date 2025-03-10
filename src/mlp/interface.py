from tkinter import *
from tkinter import ttk

def interface():

    geometry = "600x700"

    root = Tk()
    root.title("Multilayer Perceptron")
    root.geometry(geometry)
    root.configure(bg="#F0F0F0")

    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TFrame", background="#F0F0F0")

    style.configure("TLabel",
                    background="#F0F0F0",
                    foreground="#333333",
                    font=("Arial", 12))

    style.configure("Rounded.TEntry",
                    fieldbackground="white",
                    bordercolor="#CCCCCC",
                    lightcolor="#CCCCCC", 
                    foreground="#000000",
                    padding=5,
                    borderwidth=2,
                    relief="solid")

    style.layout("Rounded.TEntry",
        [
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
        ]
    )
    
    style.configure("Red.TButton",
        foreground="white",
        background="#FF0000",
        padding=5,
        borderwidth=2,
        relief="solid",
        anchor="center"
    )

    style.map("Red.TButton",
        foreground=[("active", "black")],
        background=[("active", "white")]
    )


    style.layout("Red.TButton",
        [
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
        ]
    )

    container = ttk.Frame(root)
    container.pack(expand=True, fill="both")

    frm = ttk.Frame(container, padding=10, style="TFrame")
    frm.pack(anchor="center")

    learning_rate_label = ttk.Label(frm, text="Taxa de Aprendizado:")
    learning_rate_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

    learning_rate_entry = ttk.Entry(frm, style="Rounded.TEntry", width=25, font=("Arial", 12))
    learning_rate_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    error_label = ttk.Label(frm, text="Erro Máximo Tolerado:")
    error_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    error_entry = ttk.Entry(frm, style="Rounded.TEntry", width=25, font=("Arial", 12))
    error_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    number_neurons_label = ttk.Label(frm, text="Quantidade de Neuronios:")
    number_neurons_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    number_neurons_entry = ttk.Entry(frm, style="Rounded.TEntry", width=25, font=("Arial", 12))
    number_neurons_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    sample_size_label = ttk.Label(frm, text="Tamanho da Amostra:")
    sample_size_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

    sample_size_entry = ttk.Entry(frm, style="Rounded.TEntry", width=25, font=("Arial", 12))
    sample_size_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

    x_min_label = ttk.Label(frm, text="Valor Mínimo de X:")
    x_min_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

    x_min_entry = ttk.Entry(frm, style="Rounded.TEntry", width=25, font=("Arial", 12))
    x_min_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

    x_max_label = ttk.Label(frm, text="Valor Máximo de X:")
    x_max_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")

    x_max_entry = ttk.Entry(frm, style="Rounded.TEntry", width=25, font=("Arial", 12))
    x_max_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")

    bottom_frm = ttk.Frame(container, style="TFrame")
    bottom_frm.pack(anchor="center")

    v_btn = ttk.Button(bottom_frm,
                     text="Realizar Aproximação",
                     style="Red.TButton",
                     width=20,
                     command=lambda: start_model(matrix, madaline, letter_text_label))
    v_btn.grid(row=6, column=0, columnspan=3, padx=5, pady= 10, sticky="w")

    root.mainloop()
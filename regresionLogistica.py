import os
import pickle
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from tkinter import messagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "server_failures.data")
_COLS = ["cpu_uso", "ram_uso", "latencia_ms", "temperatura_c", "procesos", "disco_uso", "clase"]

COLORES       = ["#e74c3c", "#e67e22", "#3498db", "#9b59b6"]
NOMBRES_CLASE = ["Fallo de CPU", "Fallo de RAM", "Fallo de Red", "Fallo de Disco"]
COLOR_OK      = "#2ecc71"
UMBRAL_FALLO  = 0.5   # si ningún clasificador supera este valor → sin fallo
BG_FIG  = "#16213e"
BG_AXES = "#0f1c36"

#  ENTRENAMIENTO

def _sigmoid(z):
    # Evita overflow numérico al clipear z antes de exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def _train_binario(X, y_bin, lr=0.1, epochs=3000, l2=0.01, cb=None):
    m, n = X.shape
    w = np.zeros(n)   # pesos iniciales en cero
    b = 0.0           # bias inicial en cero

    for epoch in range(1, epochs + 1):
        # Forward pass: predicción con sigmoide
        h     = _sigmoid(X.dot(w) + b)
        error = h - y_bin

        # Gradientes (descenso de gradiente + regularización L2)
        dw = (1/m) * X.T.dot(error) + (l2/m) * w
        db = (1/m) * np.sum(error)

        # Actualizar pesos y bias
        w -= lr * dw
        b -= lr * db

        if cb and epoch % 100 == 0:
            cb(epoch, epochs)

    return w, b


def entrenar_modelo(callback=None):
    """
    Carga server_failures.csv, entrena 4 clasificadores One-vs-All
    y devuelve el modelo como dict (sin guardar .pkl).
    Incluye PCA 2D para visualizar el espacio de decisión.
    """
    df = pd.read_csv(DATA_PATH, header=None, names=_COLS)
    X  = df.drop(columns=["clase"]).values.astype(float)
    y  = df["clase"].values.astype(int)

    # División 75% entrenamiento / 25% prueba
    np.random.seed(42)
    idx   = np.random.permutation(len(X))
    corte = int(len(X) * 0.75)
    X_train, X_test = X[idx[:corte]], X[idx[corte:]]
    y_train, y_test = y[idx[:corte]], y[idx[corte:]]

    # Escalado estándar (fit solo en train, evita data leakage)
    media = X_train.mean(axis=0)
    std   = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - media) / std
    X_test  = (X_test  - media) / std

    # One-vs-All: un clasificador binario por cada clase
    # Clasificador k responde "¿es clase k? Sí / No"
    clases         = [0, 1, 2, 3]
    clasificadores = []
    total_epochs   = 3000 * len(clases)
    done_epochs    = 0

    for clase in clases:
        # Convertir etiquetas a binario: 1 si es esta clase, 0 si no
        y_bin = (y_train == clase).astype(float)

        def _cb(ep, total, c=clase, d=done_epochs):
            if callback:
                callback(d + ep, total_epochs, c)

        w, b = _train_binario(X_train, y_bin, cb=_cb)
        clasificadores.append((w, b))
        done_epochs += 3000

    # Predicción: ganar la clase con mayor probabilidad (argmax)
    probs  = np.column_stack([_sigmoid(X_test.dot(w) + b) for w, b in clasificadores])
    y_pred = np.argmax(probs, axis=1)
    acc    = np.sum(y_pred == y_test) / len(y_test)
    print(f"Entrenamiento completo — Accuracy (prueba): {acc:.4f}")

    # PCA 2D para visualizar el espacio de decisión
    # SVD sobre X_train (ya centrado por el escalado estándar)
    # V_pca: las 2 direcciones de máxima varianza en los datos
    _, _, Vt = np.linalg.svd(X_train, full_matrices=False)
    V_pca      = Vt[:2].T           # (6 features, 2 componentes)
    X_train_2d = X_train @ V_pca    # proyección de entrenamiento a 2D

    modelo = {
        "clasificadores": clasificadores,
        "media":          media,
        "std":            std,
        "clases":         ["CPU", "RAM", "Red", "Disco"],
        "accuracy":       acc,
        "V_pca":          V_pca,
        "X_train_2d":     X_train_2d,
        "y_train":        y_train,
    }

    pkl_path = os.path.join(BASE_DIR, "modelo.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {pkl_path}")

    return modelo

#  APLICACIÓN

class App:
    def __init__(self, root):
        self.root   = root
        self.modelo = None
        self.root.title("Predictor de Fallos de Servidor")
        self.root.geometry("500x875")
        self.root.resizable(False, False)
        self._build_loading()
        threading.Thread(target=self._run_training, daemon=True).start()

    # ── Pantalla de carga 

    def _build_loading(self):
        self.load_frame = ctk.CTkFrame(self.root, fg_color="#1a1a2e", corner_radius=0)
        self.load_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        ctk.CTkLabel(self.load_frame, text="🖥  Predictor de Fallos",
                     font=("Arial", 22, "bold"), text_color="white").pack(pady=(140, 8))

        ctk.CTkLabel(self.load_frame, text="Entrenando clasificadores One-vs-All...",
                     font=("Arial", 11), text_color="#aaaaaa").pack(pady=4)

        self.progress = ctk.CTkProgressBar(self.load_frame, width=320)
        self.progress.set(0)
        self.progress.pack(pady=18)

        self.lbl_carga = ctk.CTkLabel(self.load_frame, text="Iniciando...",
                                      font=("Arial", 10), text_color="#555577")
        self.lbl_carga.pack()

    def _update_progress(self, done, total, clase_actual):
        pct = done / total

        def _safe_progress(p=pct):
            try:
                self.progress.set(p)
            except Exception:
                pass

        def _safe_label(c=clase_actual):
            try:
                self.lbl_carga.configure(
                    text=f"Clasificador {c+1}/4 — {NOMBRES_CLASE[c]}")
            except Exception:
                pass

        self.root.after(0, _safe_progress)
        if done % 600 == 0:
            self.root.after(0, _safe_label)

    def _run_training(self):
        try:
            self.modelo = entrenar_modelo(callback=self._update_progress)
            self.root.after(0, self._show_main)
        except FileNotFoundError:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"No se encontró:\n{DATA_PATH}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error de entrenamiento", str(e)))

    # ── Ventana principal ─────────────────────────────────────────────────────

    def _show_main(self):
        self.load_frame.destroy()
        self._build_header()
        self._build_inputs()
        self._build_button()
        self._build_result()
        self._build_plot()

    # ── Encabezado ────────────────────────────────────────────────────────────

    def _build_header(self):
        enc = ctk.CTkFrame(self.root, fg_color="#1a1a2e", corner_radius=0, height=55)
        enc.pack(fill="x")
        enc.pack_propagate(False)
        ctk.CTkLabel(enc, text="🖥  Predictor de Fallos",
                     font=("Arial", 18, "bold"), text_color="white").pack(pady=4)
        ctk.CTkLabel(enc,
                     text=f"Diagnóstico de servidores  •  Accuracy: {self.modelo['accuracy']:.1%}",
                     font=("Arial", 10), text_color="#aaaaaa").pack()

    # ── Entradas ──────────────────────────────────────────────────────────────

    def _build_inputs(self):
        frame_in = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#16213e")
        frame_in.pack(padx=25, pady=6, fill="x")
        ctk.CTkLabel(frame_in, text="Métricas del servidor",
                     font=("Arial", 12, "bold"), text_color="#aaaaaa").pack(pady=(8, 2))
        ctk.CTkFrame(frame_in, height=1, fg_color="#2a2a5a").pack(fill="x", padx=15, pady=2)

        campos = [
            ("cpu",         "⚡  CPU (%)"),
            ("ram",         "💾  RAM (%)"),
            ("latencia",    "🌐  Latencia (ms)"),
            ("temperatura", "🌡  Temperatura (°C)"),
            ("procesos",    "⚙  Procesos"),
            ("disco",       "💿  Disco (%)"),
        ]

        self.entradas = {}
        for nombre, etiqueta in campos:
            fila = ctk.CTkFrame(frame_in, fg_color="transparent")
            fila.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(fila, text=etiqueta, width=165,
                         anchor="w", font=("Arial", 11)).pack(side="left")
            e = ctk.CTkEntry(fila, width=145, height=28, corner_radius=8)
            e.pack(side="right")
            self.entradas[nombre] = e

        ctk.CTkFrame(frame_in, height=1, fg_color="#2a2a5a").pack(fill="x", padx=15, pady=5)

    # ── Botón ─────────────────────────────────────────────────────────────────

    def _build_button(self):
        ctk.CTkButton(
            self.root, text="DIAGNOSTICAR", height=38,
            font=("Arial", 13, "bold"), corner_radius=10,
            fg_color="#0f3460", hover_color="#e94560",
            command=self._diagnosticar
        ).pack(pady=5)

    # ── Resultado ─────────────────────────────────────────────────────────────

    def _build_result(self):
        frame_res = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#16213e")
        frame_res.pack(padx=25, pady=4, fill="x")
        self.lbl_resultado = ctk.CTkLabel(
            frame_res, text="Esperando diagnóstico...",
            font=("Arial", 13, "bold"), text_color="#555577"
        )
        self.lbl_resultado.pack(pady=8)

    # ── Gráfica: espacio de decisión (PCA 2D) 

    def _build_plot(self):
        frame_plot = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#16213e")
        frame_plot.pack(padx=25, pady=8, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(4.6, 2.8))
        self.fig.patch.set_facecolor(BG_FIG)
        self.ax.set_facecolor(BG_AXES)
        self.ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#2a2a5a")

        X_2d  = self.modelo["X_train_2d"]
        y_tr  = self.modelo["y_train"]
        V_pca = self.modelo["V_pca"]

        # Puntos de entrenamiento proyectados al plano PCA, coloreados por clase
        for clase in range(4):
            mask = y_tr == clase
            self.ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                            color=COLORES[clase], s=14, alpha=0.40,
                            label=NOMBRES_CLASE[clase], zorder=2)

        # Fronteras de decisión: σ(w·x + b) = 0.5  →  w·x + b = 0
        # En espacio PCA: (V.T @ w)·z + b = 0  →  z₂ = -(a₀·z₁ + b) / a₁
        z1_range = np.linspace(X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5, 300)

        for i, (w, b) in enumerate(self.modelo["clasificadores"]):
            # Proyectar el vector de pesos al espacio PCA
            a = V_pca.T @ w   # (2,) — componentes del hiperplano en 2D
            if abs(a[1]) > 1e-8:
                z2_frontera = -(a[0] * z1_range + b) / a[1]
                self.ax.plot(z1_range, z2_frontera,
                             color=COLORES[i], linestyle="--",
                             linewidth=1.5, alpha=0.85, zorder=3)

        # Limitar el eje Y para no mostrar fronteras fuera del rango de los datos
        self.ax.set_ylim(X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5)

        self.ax.set_title("Espacio de decisión (PCA 2D)", color="white", fontsize=9)
        self.ax.set_xlabel("Componente Principal 1", color="#aaaaaa", fontsize=8)
        self.ax.set_ylabel("Componente Principal 2", color="#aaaaaa", fontsize=8)

        # Leyenda solo con los puntos de cada clase (las líneas tienen el mismo color)
        self.ax.legend(fontsize=6.5, loc="best",
                       facecolor="#1a1a2e", edgecolor="#2a2a5a",
                       labelcolor="#cccccc", markerscale=1.4)

        # Nota explicativa al pie
        self.fig.text(0.5, 0.01,
                      "Frontera: σ(w·x + b) = 0.5  →  w·x + b = 0",
                      ha="center", color="#555577", fontsize=7.5)

        self.fig.tight_layout(pad=1.2, rect=[0, 0.07, 1, 1])

        self._punto_pred = None   # marcador del punto diagnosticado
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ── Función diagnosticar ──────────────────────────────────────────────────

    def _diagnosticar(self):
        try:
            # Leer y armar el vector de entrada
            X = np.array([[
                float(self.entradas["cpu"].get()),
                float(self.entradas["ram"].get()),
                float(self.entradas["latencia"].get()),
                float(self.entradas["temperatura"].get()),
                float(self.entradas["procesos"].get()),
                float(self.entradas["disco"].get()),
            ]])

            # Escalar con la misma media/std del entrenamiento → vector 1D (6,)
            X_sc = ((X - self.modelo["media"]) / self.modelo["std"])[0]

            # Calcular z y probabilidad para cada clasificador (One-vs-All)
            zs    = [float(X_sc.dot(w) + b) for w, b in self.modelo["clasificadores"]]
            probs = [float(_sigmoid(z)) for z in zs]

            # La clase ganadora es la de mayor probabilidad
            clase     = int(np.argmax(probs))
            sin_fallo = probs[clase] < UMBRAL_FALLO

            # Actualizar etiqueta de resultado
            if sin_fallo:
                self.lbl_resultado.configure(
                    text=f"OK  Servidor operando con normalidad   —   Max: {probs[clase]*100:.1f}%",
                    text_color=COLOR_OK
                )
            else:
                self.lbl_resultado.configure(
                    text=f"⚠  {NOMBRES_CLASE[clase]}   —   Confianza: {probs[clase]*100:.1f}%",
                    text_color=COLORES[clase]
                )

            # Proyectar el punto de entrada al espacio PCA y marcarlo en la gráfica
            X_2d_pred = X_sc @ self.modelo["V_pca"]   # (2,)
            color_punto = COLOR_OK if sin_fallo else COLORES[clase]

            if self._punto_pred:
                self._punto_pred.remove()

            self._punto_pred = self.ax.scatter(
                X_2d_pred[0], X_2d_pred[1],
                color=color_punto, s=220, marker="*",
                edgecolors="white", linewidths=0.8, zorder=5
            )

            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Error", "Ingresá solo números en los campos")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ============

if __name__ == "__main__":
    root = ctk.CTk()
    App(root)
    root.mainloop()

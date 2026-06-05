import customtkinter as ctk
import numpy as np
from model import RegresionLogisticaProxy

# Configuración del entorno visual
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AppMonitorProxy(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("TelcoBolivia S.A. - Control en Tiempo Real Multivariable")
        self.geometry("600x750")
        self.resizable(False, False)

        # Cargar los parámetros calculados por el Gradiente Descendiente
        if not self.cargar_parametros_guardados():
            self.destroy()
            return

        self.crear_componentes()

    def cargar_parametros_guardados(self):
        """Lee el archivo de texto generado por train.py para configurar las matrices"""
        try:
            with open('pesos_modelo.txt', 'r') as f:
                lineas = f.read().splitlines()
            
            # Reconstrucción de vectores y sesgo
            self.bias_guardado = float(lineas[0])
            self.weights_guardados = np.array([float(x) for x in lineas[1:5]])
            self.X_min = np.array([float(x) for x in lineas[5:9]])
            self.X_max = np.array([float(x) for x in lineas[9:13]])
            
            # Instanciar el modelo matemático
            self.modelo = RegresionLogisticaProxy()
            self.modelo.weights = self.weights_guardados
            self.modelo.bias = self.bias_guardado
            return True
        except FileNotFoundError:
            print("\n" + "!"*60)
            print("⚠️ [ERROR CRÍTICO] No se encontró el archivo 'pesos_modelo.txt'.")
            print("Por favor, ejecuta primero la Opción [2] del menú para entrenar el modelo.")
            print("!"*60 + "\n")
            return False
        except Exception as e:
            print(f"Error al inicializar las matrices de control: {e}")
            return False

    def crear_componentes(self):
        self.lbl_titulo = ctk.CTkLabel(
            self, 
            text="PANEL DE INFERENCIA EN VIVO - 4 VARIABLES", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.lbl_titulo.pack(pady=15)

        # Generador automático de tarjetas de telemetría
        def crear_tarjeta_slider(label_text, min_v, max_v, default_v, unit):
            frame = ctk.CTkFrame(self)
            frame.pack(pady=8, padx=30, fill="x")
            lbl = ctk.CTkLabel(frame, text=label_text, font=ctk.CTkFont(size=13))
            lbl.pack(anchor="w", padx=15, pady=(8, 0))
            slider = ctk.CTkSlider(frame, from_=min_v, to=max_v, command=self.actualizar_inferencia)
            slider.set(default_v)
            slider.pack(fill="x", padx=15, pady=5)
            val_lbl = ctk.CTkLabel(frame, text=f"{default_v} {unit}", font=ctk.CTkFont(size=11, weight="bold"))
            val_lbl.pack(anchor="e", padx=15, pady=(0, 8))
            return slider, val_lbl, unit

        # Instanciar los 4 componentes mecánicos y de tráfico
        self.s_cpu, self.l_cpu, self.u_cpu = crear_tarjeta_slider("Uso de CPU:", 10.0, 100.0, 50.0, "%")
        self.s_tmp, self.l_tmp, self.u_tmp = crear_tarjeta_slider("Temperatura de Núcleos:", 40.0, 95.0, 60.0, "°C")
        self.s_req, self.l_req, self.u_req = crear_tarjeta_slider("Peticiones concurrentes de red:", 100, 5000, 1500, "req/s")
        self.s_pwr, self.l_pwr, self.u_pwr = crear_tarjeta_slider("Consumo Eléctrico de Fuente:", 50.0, 250.0, 110.0, "Watts")

        self.lbl_resultado_titulo = ctk.CTkLabel(self, text="PROBABILIDAD MATEMÁTICA DE FALLO (SIGMOIDE)", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_resultado_titulo.pack(pady=(20, 0))

        self.lbl_probabilidad = ctk.CTkLabel(self, text="0.00%", font=ctk.CTkFont(size=44, weight="bold"), text_color="#1f538d")
        self.lbl_probabilidad.pack(pady=5)

        self.card_alerta = ctk.CTkLabel(
            self, text="✅ ESTADO OPERATIVO SEGURO", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2e7d32", text_color="white", corner_radius=8, height=45, width=420
        )
        self.card_alerta.pack(pady=15)
        
        # Ejecutar la primera inferencia con los valores por defecto
        self.actualizar_inferencia()

    def actualizar_inferencia(self, *args):
        # 1. Obtener valores actuales de los sliders
        inputs = np.array([self.s_cpu.get(), self.s_tmp.get(), self.s_req.get(), self.s_pwr.get()])
        
        # Actualizar textos dinámicos de las etiquetas
        self.l_cpu.configure(text=f"{inputs[0]:.1f} {self.u_cpu}")
        self.l_tmp.configure(text=f"{inputs[1]:.1f} {self.u_tmp}")
        self.l_req.configure(text=f"{int(inputs[2])} {self.u_req}")
        self.l_pwr.configure(text=f"{inputs[3]:.1f} {self.u_pwr}")

        # 2. Ingeniería de Características: Escalado Min-Max idéntico al entrenamiento
        entrada_escalada = (inputs - self.X_min) / (self.X_max - self.X_min + 1e-10)
        entrada_escalada = entrada_escalada.reshape(1, -1)

        # 3. Forward Pass: Inferencia con la sigmoide nativa del modelo
        probabilidad = self.modelo.predict_proba(entrada_escalada)[0]
        self.lbl_probabilidad.configure(text=f"{probabilidad * 100:.2f}%")

        # 4. Control Lógico del Sistema de Alerta Temprana (SAT)
        if probabilidad >= 0.80:
            self.lbl_probabilidad.configure(text_color="#c62828")
            self.card_alerta.configure(text="🚨 ALERTA CRÍTICA: ACTIVANDO FAILOVER AUTOMÁTICO", fg_color="#c62828")
        elif probabilidad >= 0.50:
            self.lbl_probabilidad.configure(text_color="#f57c00")
            self.card_alerta.configure(text="⚠️ ADVERTENCIA: HARDWARE BAJO COMPRESIÓN", fg_color="#f57c00")
        else:
            self.lbl_probabilidad.configure(text_color="#2e7d32")
            self.card_alerta.configure(text="✅ ESTADO OPERATIVO SEGURO", fg_color="#2e7d32")

if __name__ == "__main__":
    app = AppMonitorProxy()
    app.mainloop()
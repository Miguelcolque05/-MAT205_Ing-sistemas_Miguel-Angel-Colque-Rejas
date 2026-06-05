import os
import sys

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_menu():
    limpiar_pantalla()
    print("=" * 60)
    print("   SISTEMA PREDICTIVO DE FALLOS - TELCOBOLIVIA .   ")
    print("         LABORATORIO DE MÉTODOS NUMÉRICOS             ")
    print("=" * 60)
    print(" [1] Generar / Resetear Dataset Multivariable ")
    print(" [2] Entrenar Modelo, Ver Pérdida y Matriz de Confusión")
    print(" [3] Visualizar Curva de Aprendizaje (Pérdida vs Accuracy)")
    print(" [4] Ver Proyección de Servidores en la Función Sigmoide")
    print(" [5] Lanzar Panel de Control en Tiempo Real (Interfaz )")
    print(" [6] Salir del Sistema")
    print("=" * 60)

def ejecutar_opcion(opcion):
    if opcion == "1":
        print("\n[PROCESO] Generando matriz de telemetría...")
        if 'generar_data' in sys.modules:
            del sys.modules['generar_data']
        import generar_data
        print("\nPresiona Enter para volver al menú...")
        input()
        
    elif opcion == "2":
        print("\n[PROCESO] Iniciando optimización por Gradiente Descendiente...")
        if 'train' in sys.modules:
            del sys.modules['train']
        import train
        print("\nPresiona Enter para volver al menú...")
        input()
        
    elif opcion == "3":
        print("\n[PROCESO] Renderizando curva de aprendizaje...")
        print("[INFO] Cierra la ventana del gráfico para regresar al menú.")
        os.system('py curva_aprendizaje.py')
        print("\nPresiona Enter para volver al menú...")
        input()

    elif opcion == "4":
        print("\n[PROCESO] Proyectando estados de servidores sobre la Sigmoide...")
        print("[INFO] Cierra la ventana del gráfico para regresar al menú.")
        os.system('py grafico_sigmoide.py')
        print("\nPresiona Enter para volver al menú...")
        input()

    elif opcion == "5":
        print("\n[PROCESO] Inicializando componentes visuales de CustomTkinter...")
        print("[INFO] Carga completada. Cierra la ventana de la GUI para regresar al menú.")
        # Forzamos la ejecución de la GUI en un proceso independiente para evitar congelamientos
        os.system('py gui.py')
        print("\nPresiona Enter para volver al menú...")
        input()
        
    elif opcion == "6":
        print("\nSaliendo del sistema ")
        sys.exit()
    else:
        print("\n⚠️ Opción no válida. Intenta de nuevo.")
        os.system('timeout /t 2 >nul' if os.name == 'nt' else 'sleep 2')

def main():
    while True:
        mostrar_menu()
        eleccion = input("Seleccione una opción (1-6): ").strip()
        ejecutar_opcion(eleccion)

if __name__ == "__main__":
    main()
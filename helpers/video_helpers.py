import os
import cv2

def dfi_video(frames_output_folder):
    # Directorio de entrada de imágenes
    ruta_entrada = frames_output_folder
    
    # Obtener el tamaño de la primera imagen en el directorio de entrada
    img_path = os.path.join(ruta_entrada, os.listdir(ruta_entrada)[0])
    img = cv2.imread(img_path)
    img_size = (img.shape[1], img.shape[0])
    
    # Fps del video
    fps = 15
    
    # Crear objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_salida = cv2.VideoWriter('output.mp4', fourcc, fps, img_size)
    
    # Crear ventana con nombre "video"
    cv2.namedWindow("video")
    
    # Establecer la ventana en primer plano
    cv2.setWindowProperty("video", cv2.WND_PROP_TOPMOST,1.0)

    # Crear ventana de visualización
    # Leer imágenes en el directorio y agregarlas al video de salida
    for file in sorted(os.listdir(ruta_entrada)):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"): # Verificar que sea una imagen
            img = cv2.imread(os.path.join(ruta_entrada, file)) # Leer la imagen
            #img_resized = cv2.resize(img, img_size) # Redimensionar la imagen
            video_salida.write(img) # Agregar la imagen al video
            
    # Liberar el objeto VideoWriter
    video_salida.release()
    
    # Crear objeto VideoCapture para leer el archivo de video recién creado
    video_capture = cv2.VideoCapture('output.mp4')
    
    # Crear ventana con nombre "video"
    cv2.namedWindow("video")
    
    # Establecer la ventana en primer plano
    cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    # Mostrar el video en una ventana
    while True:
        ret, img = video_capture.read()
        if ret:
            cv2.imshow('video', img)
            cv2.waitKey(int(1000/fps))
        else:
            break
    
    # Liberar el objeto VideoCapture y cerrar la ventana de visualización
    video_capture.release()
    cv2.destroyAllWindows()
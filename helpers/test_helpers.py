import os
import cv2
from tqdm import tqdm

from helpers.constant_helpers import RUN_FOLDER, init_project_folders
from helpers.file_helpers import copy_images
from helpers.image_helpers import blur_all_images, denoise, dilate_all_images, greyscale_all_images, sresize

maskDT = f"{RUN_FOLDER}/MaskDT"
maskST = f"{RUN_FOLDER}/MaskST"
sourceT = f"{RUN_FOLDER}/SourceT"
outputT = f"{RUN_FOLDER}/OutputT"
genT = f"{RUN_FOLDER}/GenT"


def test_dfi(reference_frames_folder, generated_frames_folder, denoise_blur, dfi_strength, dfi_deghost, test_mode, smooth):
    (maskD, maskS, output, source, gen) = init_project_folders(
        maskD_override=maskDT,
        maskS_override=maskST,
        output_override=outputT,
        source_override=sourceT,
        gen_override=genT,
    )

    # Usar el primer formato
    indices = [10, 11, 20, 21, 30, 31] # Los índices de las imágenes que quieres copiar
    
    if test_mode != 0:
        test_frames = test_mode
        # Usar el segundo formato
        indices = list(range(test_frames)) # Los primeros 30 índices
                
    # Llamar a la función copy_images para copiar las imágenes
    copy_images(reference_frames_folder, destination=source, frame_indicies=indices)
    sresize(generated_frames_folder, source)
    denoise(denoise_blur, source)

    greyscale_all_images(source, maskD, dfi_strength)
    dilate_all_images(maskD, dfi_deghost)
    blur_all_images(maskD, smooth)

                
    if test_mode == 0:         
        nombres = os.listdir(maskD) # obtener los nombres de los archivos en la carpeta MaskDT
        ancho = 0 # variable para guardar el ancho acumulado de las ventanas
        for i, nombre in tqdm(enumerate(nombres), total=len(nombres), desc="Iterate On Test Window"): # recorrer cada nombre de archivo
            imagen = cv2.imread(f"{maskD}/" + nombre) # leer la imagen correspondiente
            h, w, c = imagen.shape # obtener el alto, ancho y canales de la imagen
            aspect_ratio = w / h # calcular la relación de aspecto
            cv2.namedWindow(nombre, cv2.WINDOW_NORMAL) # crear una ventana con el nombre del archivo
            ancho_ventana = 630 # definir un ancho fijo para las ventanas
            alto_ventana = int(ancho_ventana / aspect_ratio) # calcular el alto proporcional al ancho y a la relación de aspecto
            cv2.resizeWindow(nombre, ancho_ventana, alto_ventana) # cambiar el tamaño de la ventana según las dimensiones calculadas 
            cv2.moveWindow(nombre, ancho, 0) # mover la ventana a una posición horizontal según el ancho acumulado 
            cv2.imshow(nombre, imagen) # mostrar la imagen en la ventana
            cv2.setWindowProperty(nombre,cv2.WND_PROP_TOPMOST,1.0) # poner la ventana en primer plano con un valor double 
            ancho += ancho_ventana + 10 # aumentar el ancho acumulado en 410 píxeles para la siguiente ventana 
        cv2.waitKey(4000) # esperar a que se presione una tecla para cerrar todas las ventanas 
        cv2.destroyAllWindows() # cerrar todas las ventanas abiertas por OpenCV 
    else:
        # Directorio de entrada de imágenes
        ruta_entrada = maskD
        
        # Obtener el tamaño de la primera imagen en el directorio de entrada
        img_path = os.path.join(ruta_entrada, os.listdir(ruta_entrada)[0])
        img = cv2.imread(img_path)
        img_size = (img.shape[1], img.shape[0])
        
        # Fps del video
        fps = 10
        
        # Crear objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_salida = cv2.VideoWriter('output.mp4', fourcc, fps, img_size)
        
        # Crear ventana con nombre "video"
        cv2.namedWindow("video")
        
        # Establecer la ventana en primer plano
        cv2.setWindowProperty("video", cv2.WND_PROP_TOPMOST,1.0)

        # Crear ventana de visualización
        # Leer imágenes en el directorio y agregarlas al video de salida
        for file in tqdm(sorted(os.listdir(ruta_entrada)), desc="Creating Video"):
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


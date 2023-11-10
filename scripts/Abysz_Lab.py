import os
import cv2
import shutil
from tqdm import tqdm

from modules import shared
from modules import scripts
from modules import script_callbacks

from helpers.file_helpers import copy_images, copy_rename_images
from helpers.image_helpers import blur_all_images, denoise, dilate_all_images, dyndef, greyscale_all_images, main_process_loop, normalize_deflicker, over_fuse, overlay_deflicker, overlay_images, sresize
from helpers.constant_helpers import destroy_project_folders, init_deflicker_folders, init_project_folders
from helpers.grado_helpers import add_tab
from helpers.test_helpers import test_dfi
from helpers.video_helpers import dfi_video

class Script(scripts.Script):
    def title(self):
        return "Abysz LAB"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
     
    def ui(self, is_img2img):
        return []
        
def main(
    reference_frames_folder, 
    generated_frames_folder, 
    frames_output_folder, 
    denoise_blur, 
    dfi_strength, 
    dfi_deghost, 
    test_mode, 
    inter_denoise, 
    inter_denoise_size, 
    inter_denoise_speed, 
    fine_blur, 
    frame_refresh_frequency, 
    refresh_strength, 
    smooth, 
    frames_limit,
    magick_file_location_override,
):
    # ensure a clean project
    destroy_project_folders()
    (maskD, maskS, output, source, gen) = init_project_folders()
    
    copy_images(reference_frames_folder, source_folder=source, frames_limit=frames_limit)
    
    sresize(generated_frames_folder, source_folder=source)
    
    copy_rename_images(generated_frames_folder, gen)
   
    denoise(denoise_blur, source_folder=source)

    greyscale_all_images(source, maskD, dfi_strength)
    dilate_all_images(maskD, dfi_deghost)
    blur_all_images(maskD, smooth)
    
    main_process_loop(
        maskD=maskD,
        maskS=maskS,
        source=source,
        gen=gen,
        frames_output_folder=frames_output_folder,
        fine_blur=fine_blur,
        frame_refresh_frequency=frame_refresh_frequency,
        refresh_strength=refresh_strength,
        inter_denoise=inter_denoise,
        inter_denoise_size=inter_denoise_size,
        inter_denoise_speed=inter_denoise_speed,
        magick_file_location_override=magick_file_location_override,
    )

        
def deflickers(deflicker_frames_folder, deflicker_frames_output_folder, ddf_strength, over_strength, norm_strength):
    (overlay_folder, _) = init_deflicker_folders()
    dyndef(
        deflicker_frames_folder, 
        deflicker_frames_output_folder, 
        ddf_strength
    )
    overlay_deflicker(
        deflicker_frames_folder, 
        deflicker_frames_output_folder, 
        ddf_strength, 
        over_strength,
        overlay_folder,
    )
    normalize_deflicker(
        deflicker_frames_folder,
        deflicker_frames_output_folder,
        ddf_strength, 
        over_strength,
        norm_strength
    )


def extract_video(video_extract_original_folder, video_extract_output_folder, fps_count):
    # Ruta del archivo de video
    filename = video_extract_original_folder

    # Directorio donde se guardarán los frames extraídos
    output_dir = video_extract_output_folder

    # Abrir el archivo de video
    cap = cv2.VideoCapture(filename)

    # Obtener los FPS originales del video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Si fps_count es 0, utilizar los FPS originales
    if fps_count == 0:
        fps_count = fps

    # Calcular el tiempo entre cada frame a extraer en milisegundos
    frame_time = int(round(1000 / fps_count))

    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Inicializar el contador de frames
    frame_count = 0

    # Inicializar el tiempo del último frame extraído
    last_frame_time = 0

    # Iterar sobre los frames del video
    while True:
        # Leer el siguiente frame
        ret, frame = cap.read()

        # Si no se pudo leer un frame, salir del loop
        if not ret:
            break

        # Calcular el tiempo actual del frame en milisegundos
        current_frame_time = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Si todavía no ha pasado suficiente tiempo desde el último frame extraído, saltar al siguiente frame
        if current_frame_time - last_frame_time < frame_time:
            continue

        # Incrementar el contador de frames
        frame_count += 1

        # Construir el nombre del archivo de salida
        output_filename = os.path.join(output_dir, 'frame_{:05d}.jpeg'.format(frame_count))

        # Guardar el frame como una imagen
        cv2.imwrite(output_filename, frame)

        # Actualizar el tiempo del último frame extraído
        last_frame_time = current_frame_time

    # Cerrar el archivo de video
    cap.release()

    # Mostrar información sobre el proceso finalizado
    print("Extracted {} frames.".format(frame_count))


def on_ui_tabs_fn():
    return add_tab(
        main_fn=main,
        deflickers_fn=deflickers,
        extract_video_fn=extract_video,
        dfi_video_fn=dfi_video,
        over_fuse_fn=over_fuse,
        test_dfi_fn=test_dfi,
    )


script_callbacks.on_ui_tabs(on_ui_tabs_fn)

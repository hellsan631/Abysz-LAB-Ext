import os
import numpy as np
from PIL import Image
import cv2
import shutil
import math
from tqdm import tqdm

from helpers.magick_helpers import get_magic_command
from helpers.system_helpers import run_system_command

def sresize(reference_frames_folder, source_folder):
    gen_folder = reference_frames_folder
    
    # Get the first image in the Gen folder
    gen_images = os.listdir(gen_folder)
    gen_image_path = os.path.join(gen_folder, gen_images[0])
    gen_image = cv2.imread(gen_image_path)
    gen_height, gen_width = gen_image.shape[:2]
    gen_aspect_ratio = gen_width / gen_height
    
    # Iterate through all the FULL images
    for image_name in tqdm(sorted(os.listdir(source_folder)), desc="Resizing Source Images"): 
        image_path = os.path.join(source_folder, image_name)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        aspect_ratio = width / height
    
        if aspect_ratio != gen_aspect_ratio:
            if aspect_ratio > gen_aspect_ratio:
                # The image is wider than Gen's image
                crop_width = int(height * gen_aspect_ratio)
                x = int((width - crop_width) / 2)
                image = image[:, x:x+crop_width]
            else:
                # The image is taller or equal to Gen's image
                crop_height = int(width / gen_aspect_ratio)
                y = int((height - crop_height) / 2)
                image = image[y:y+crop_height, :]
    
        # Resize image from FULL to Gen image resolution
        image = cv2.resize(image, (gen_width, gen_height))
    
        # Save the resized image in the FULL folder
        cv2.imwrite(os.path.join(source_folder, image_name), image)


def denoise(denoise_blur, source_folder):
    # Condition 1: strength must be greater than 1
    if denoise_blur < 1: 
        return

    # Get the list of file names in the source folder
    files = os.listdir(source_folder)
    
    # Iterate through all the images in the source folder
    for file in tqdm(files, desc="Denoising Source Images"):
        # Read the image with opencv
        img = cv2.imread(os.path.join(source_folder, file))
    
        # Apply the blur filter with a 5x5 kernel size
        dst = cv2.bilateralFilter(img, denoise_blur, 31, 31)
    
        # Save the resulting image in the destination folder with the same name
        cv2.imwrite(os.path.join(source_folder, file), dst)


def greyscale_all_images(
    source,
    maskD,
    dfi_strength,
):
    count = 1
    sorted_list = sorted(os.listdir(source))
    previous_image = None
    # Iterate through the image files in the Source source
    for filename in tqdm(sorted_list, desc="Greyscaling Source Images"):
        # Load current and next image in grayscale
        if previous_image is not None:
            next_image = cv2.imread(os.path.join(source, filename), cv2.IMREAD_GRAYSCALE)
            diff = cv2.absdiff(previous_image, next_image)

            # Apply a threshold and save the resulting image to the MaskD source. Less is more.
            thresholded_file = cv2.threshold(diff, dfi_strength, 255, cv2.THRESH_BINARY_INV)[1] # Intrevert Colors
            cv2.imwrite(os.path.join(maskD, f'{count-1:04d}.png'), thresholded_file)
    
        previous_image = cv2.imread(os.path.join(source, filename), cv2.IMREAD_GRAYSCALE)

        # Currently, the thresholding type is cv2.THRESH_BINARY_INV, which inverts the colors of the thresholded image.
        # You can change it to another type of thresholding,
        # with cv2.THRESH_BINARY, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO o cv2.THRESH_TOZERO_INV.


def dilate_all_images(
    maskD,
    dfi_deghost,
):
    if dfi_deghost == 0:
        return
    
    files = os.listdir(maskD)
    
    for file in tqdm(files, desc="Dilating MaskD Images"):
        img = cv2.imread(os.path.join(maskD, file))
        
        # Invert the image using the bitwise_not() function
        img_inv = cv2.bitwise_not(img)
        
        kernel_size = dfi_deghost
        
        # Dilate the image using the dilate() function
        # You can change the size and shape of the kernel according to your preferences
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)) 
        img_dil = cv2.dilate(img_inv, kernel)
        
        # Reverse the image using the bitwise_not() function
        img_out = cv2.bitwise_not(img_dil)
        
        # Overwrite the image in the MaskD folder with the same name as the original
        filename = os.path.join(maskD, file)
        cv2.imwrite(filename, img_out)


def blur_all_images(
    maskD,
    smooth,
):
    has_blur_kernel = smooth > 1
    if not has_blur_kernel:
        return

    for imagen in tqdm(os.listdir(maskD), desc="Blurring MaskD Images"):
        if imagen.endswith(".jpg") or imagen.endswith(".png") or imagen.endswith(".jpeg"):
            img = cv2.imread(os.path.join(maskD, imagen))
            # Apply filter
            img = cv2.GaussianBlur(img, (smooth,smooth),0)
            # Save the image with the same name
            cv2.imwrite(os.path.join(maskD, imagen), img)


def main_process_loop(
    maskD,
    maskS,
    source,
    gen,
    frames_output_folder,
    fine_blur,
    frame_refresh_frequency,
    refresh_strength,
    inter_denoise_speed,
    inter_denoise,
    inter_denoise_size,
    magick_file_location_override = None,
):
    # START BATCH Get the file name in MaskD without any extension
    # Add a loop counter variable
    pbar = tqdm(total=None, desc='Running Main Process Loop', unit=' ops', leave=True)
    loop_count = 0

    # Ensure that we have a file in the Gen folder before beginning the loop
    gen_files = os.listdir(gen)
    if gen_files:
        first_gen_file = gen_files[0]
        output_file = os.path.join(frames_output_folder, first_gen_file)
        shutil.copyfile(os.path.join(gen, first_gen_file), output_file)

    # Add a while loop to run the code in an infinite loop
    while True:
        mask_files = sorted(os.listdir(maskD))
        if not mask_files:
            print(f"No frames left")
            # Delete the Source, MaskS, and MaskD folders if there are no more files to process
            shutil.rmtree(maskD)
            shutil.rmtree(maskS)
            shutil.rmtree(source)
            break
        
        extra_mod = fine_blur
        
        mask = mask_files[0]
        maskname = os.path.splitext(mask)[0]
        
        maskp_path = os.path.join(maskD, mask) 

        img = cv2.imread(maskp_path, cv2.IMREAD_GRAYSCALE) # read the image in grayscale
        n_white_pix = np.sum(img == 255) # count the pixels that are equal to 255 (white)
        total_pix = img.size # get the total number of pixels in the image
        percentage = (n_white_pix / total_pix) * 100 # calculate the percentage of white pixels
        percentage = round(percentage, 1) # round the percentage to 1 decimal
        
        # calculate the extra variable
        extra = 100 - percentage # subtract the percentage from 100
        extra = extra / 3 # divide the result by 3
        extra = math.ceil(extra) # round up to the nearest whole number
        if extra % 2 == 0: # check if the number is even
            extra = extra + 1 # add 1 to make it odd
        
        # Dynamic Blur
        imgb = cv2.imread(maskp_path) # read the image with opencv
        img_blur = cv2.GaussianBlur(imgb, (extra,extra),0)

        # save the modified image with the same name and path
        cv2.imwrite(maskp_path, img_blur)

        all_output_files = os.listdir(frames_output_folder)

        # Get the path of the image in the output subfolder that has the same name as the image in MaskD
        output_files = [file for file in all_output_files if os.path.splitext(file)[0] == maskname]
        
        if not output_files:
            print(f"No image found in {frames_output_folder} with the same name as {maskname}.")
            print(f"All files in {frames_output_folder}: {all_output_files}")
            exit(1)
        
        output_file = os.path.join(frames_output_folder, output_files[0])
        
        # Apply the magick composite command with the desired options
        compose_command = f"composite -compose CopyOpacity {os.path.join(maskD, mask)} {output_file} {os.path.join(maskS, 'result.png')}"
        run_system_command(get_magic_command(compose_command), magick_file_location_override)
        
        # Get the name of the file in output without any extension
        name = os.path.splitext(os.path.basename(output_file))[0]
        
        # Rename the result.png file with the name of the file in output and the .png extension
        os.rename(os.path.join(maskS, 'result.png'), os.path.join(maskS, f"{name}.png"))
        
        # Save the current directory in a variable
        original_dir = os.getcwd()
        
        # Change to the directory of the MaskS folder
        os.chdir(maskS)
        
        # Iterate through the image files in the MaskS folder
        for image in tqdm(sorted(os.listdir(".")), desc="Renaming MaskS Images"):
            # Get the name of the image without the extension
            name, extension = os.path.splitext(image)
            # Get only the number of the image
            number = ''.join(filter(str.isdigit, name))
            # Define the name of the next image
            next_name = f"{int(number)+1:0{len(number)}}{extension}"
            # Rename the image
            os.rename(image, next_name)
        
        # Return to the original directory
        os.chdir(original_dir)
        
        # Set a default value for dissolution
        if frame_refresh_frequency < 1:
            dissolve = percentage
        else:
            dissolve = 100 if loop_count % frame_refresh_frequency != 0 else refresh_strength
                
        
        # Get the name of the file in MaskS without the extension
        maskS_files = [f for f in os.listdir(maskS) if os.path.isfile(os.path.join(maskS, f)) and f.endswith('.png')]
        if maskS_files:
            filename = os.path.splitext(maskS_files[0])[0]
        else:
            print(f"No image files found in the folder '{maskS}'")
            filename = ''[0]
        
        # Exit the loop if there are no more images to process
        if not filename:
            break
        
        # Get the extension of the file in Gen with the same name
        gen_files = [f for f in os.listdir(gen) if os.path.isfile(os.path.join(gen, f)) and f.startswith(filename)]
        if gen_files:
            ext = os.path.splitext(gen_files[0])[1]
        else:
            print(f"No file found with the name '{filename}' in the folder '{gen}'")
            ext = ''
                            
        # Composite the image from MaskS and Gen with dissolution (if defined) and save it in the output folder
        composite_command = f"composite {'-dissolve ' + str(dissolve) + '%' if dissolve is not None else ''} {maskS}/{filename}.png {gen}/{filename}{ext} {frames_output_folder}/{filename}{ext}"
        run_system_command(get_magic_command(composite_command, magick_file_location_override))
        
        denoise_loop = inter_denoise_speed
        kernel1 = inter_denoise
        kernel2 = inter_denoise_size
        
        # Demo plus bilateral
        if loop_count % denoise_loop == 0:
            # list files in the output folder
            files = os.listdir(frames_output_folder)
            # get the last file
            last_file = os.path.join(frames_output_folder, files[-1])
            # load image with opencv
            image = cv2.imread(last_file)
            # apply bilateral filter
            filtered_image = cv2.bilateralFilter(image, kernel1, kernel2, kernel2)
            # overwrite the original
            cv2.imwrite(last_file, filtered_image)
        
        # Get the name of the lowest file in the MaskD folder
        maskd_files = [f for f in os.listdir(maskD) if os.path.isfile(os.path.join(maskD, f)) and f.startswith('')]
        if maskd_files:
            maskd_file = os.path.join(maskD, sorted(maskd_files)[0])
            os.remove(maskd_file)
        
        # Get the name of the lowest file in the MaskS folder
        masks_files = [f for f in os.listdir(maskS) if os.path.isfile(os.path.join(maskS, f)) and f.startswith('')]
        if masks_files:
            masks_file = os.path.join(maskS, sorted(masks_files)[0])
            os.remove(masks_file)
                            
        # Increase the loop counter
        loop_count += 1
        pbar.set_description(f'Processed {loop_count} operations')
        pbar.update(1)


def overlay_images(
    image1_path, 
    image2_path, 
    over_strength,
    prefered_image = "image1",
):
    opacity = over_strength
    
    # Abrir las imágenes
    image1 = Image.open(image1_path).convert('RGBA')
    image2 = Image.open(image2_path).convert('RGBA')

    # Alinear el tamaño de las imágenes
    if image1.size != image2.size and prefered_image == "image1":
        image2 = image2.resize(image1.size)
    elif image1.size != image2.size and prefered_image == "image2":
        image1 = image1.resize(image2.size)

    # Convertir las imágenes en matrices NumPy
    np_image1 = np.array(image1).astype(np.float64) / 255.0
    np_image2 = np.array(image2).astype(np.float64) / 255.0

    # Aplicar el método de fusión "overlay" a las imágenes
    def basic(target, blend, opacity):
        return target * opacity + blend * (1-opacity)

    def blender(func):
        def blend(target, blend, opacity=1, *args):
            res = func(target, blend, *args)
            res = basic(res, blend, opacity)
            return np.clip(res, 0, 1)
        return blend

    class Blend:
        @classmethod
        def method(cls, name):
            return getattr(cls, name)

        normal = basic

        @staticmethod
        @blender
        def overlay(target, blend, *args):
            return  (target>0.5) * (1-(2-2*target)*(1-blend)) +\
                    (target<=0.5) * (2*target*blend)

    blended_image = Blend.overlay(np_image1, np_image2, opacity)

    # Convertir la matriz de vuelta a una imagen PIL
    blended_image = Image.fromarray((blended_image * 255).astype(np.uint8), 'RGBA').convert('RGB')

    # Guardar la imagen resultante
    return blended_image


def over_fuse(fuse_style_frames_folder, fuse_video_frames_folder, fuse_output_frames_folder, fuse_strength):
    # Obtener una lista de todos los archivos en la carpeta "Gen"
    gen_files = os.listdir(fuse_style_frames_folder)
    
    # Ordenar la lista de archivos alfabéticamente
    gen_files.sort()
    
    # Obtener una lista de todos los archivos en la carpeta "Source"
    source_files = os.listdir(fuse_video_frames_folder)
    
    # Ordenar la lista de archivos alfabéticamente
    source_files.sort()
    
    if not os.path.exists(fuse_output_frames_folder):
        os.makedirs(fuse_output_frames_folder)
            
    for i in tqdm(range(len(gen_files)), desc="Fusing frames"):
        image1_path = os.path.join(fuse_style_frames_folder, gen_files[i])
        image2_path = os.path.join(fuse_video_frames_folder, source_files[i])
        blended_image = overlay_images(image1_path, image2_path, fuse_strength, "image2")
        try:
            blended_image.save(os.path.join(fuse_output_frames_folder, gen_files[i]))
        except Exception as e:
            print("Error al guardar la imagen:", str(e))
            print("No more frames to fuse")
            break


def overlay_deflicker(deflicker_frames_folder, deflicker_frames_output_folder, ddf_strength, over_strength):
    if over_strength <= 0: # Condición 1: strength debe ser mayor a 0
        return
       
    # Si ddf_strength y/o over_strength son mayores a 0, utilizar deflicker_frames_output_folder en lugar de deflicker_frames_folder
    if ddf_strength > 0:
        deflicker_frames_folder = deflicker_frames_output_folder    
        
    if not os.path.exists("overtemp"):
        os.makedirs("overtemp")
            
    if not os.path.exists(deflicker_frames_output_folder):
        os.makedirs(deflicker_frames_output_folder)
            
    gen_path = deflicker_frames_folder
    images = sorted(os.listdir(gen_path))
    image1_path = os.path.join(gen_path, images[0])
    image2_path = os.path.join(gen_path, images[1])
     
    fused_image = overlay_images(image1_path, image2_path, over_strength)
    fuseover_path = "overtemp"
    filename = os.path.basename(image1_path)
    fused_image.save(os.path.join(fuseover_path, filename))
    
    # Obtener una lista de todos los archivos en la carpeta "Gen"
    gen_files = sorted(os.listdir(deflicker_frames_folder))
    
    for i in tqdm(range(len(gen_files) - 1), desc="Overlaying frames"):
        image1_path = os.path.join(deflicker_frames_folder, gen_files[i])
        image2_path = os.path.join(deflicker_frames_folder, gen_files[i+1])
        blended_image = overlay_images(image1_path, image2_path, over_strength)
        blended_image.save(os.path.join("overtemp", gen_files[i+1]))
    
    # Definimos la ruta de la carpeta "overtemp"
    ruta_overtemp = "overtemp"
    
    for archivo in tqdm(os.listdir(ruta_overtemp) , desc="Moving Over Overlaying frames"):
        origen = os.path.join(ruta_overtemp, archivo)
        destino = os.path.join(deflicker_frames_output_folder, archivo)
        shutil.move(origen, destino)
        
    # Ajustar contraste y brillo para cada imagen en la carpeta de entrada
    if over_strength >= 0.4:
        for nombre_archivo in tdem(os.listdir(deflicker_frames_output_folder), desc="Adjusting contrast and brightness"):
            # Cargar imagen
            ruta_archivo = os.path.join(deflicker_frames_output_folder, nombre_archivo)
            img = cv2.imread(ruta_archivo)
        
            # Ajustar contraste y brillo
            alpha = 1  # Factor de contraste (mayor que 1 para aumentar el contraste)
            beta = 10  # Valor de brillo (entero positivo para aumentar el brillo)
            img_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
            # Guardar imagen resultante en la carpeta de salida
            frames_output_folder = os.path.join(deflicker_frames_output_folder, nombre_archivo)
            cv2.imwrite(frames_output_folder, img_contrast)


def normalize_frame(
    image1_path,
    image2_path,
    destination_filename,
):
    # Cargar las dos imágenes a fusionar
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Calcular la luminosidad promedio de cada imagen
    avg1 = np.mean(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    avg2 = np.mean(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

    # Calcular los pesos para cada imagen
    weight1 = avg1 / (avg1 + avg2)
    weight2 = avg2 / (avg1 + avg2)

    # Fusionar las imágenes utilizando los pesos
    result = cv2.addWeighted(img1, weight1, img2, weight2, 0)
                            
    # Guardar la imagen resultante en la carpeta GenOverNorm con el mismo nombre que la imagen original
    cv2.imwrite(os.path.join("normtemp", destination_filename), result)


def normalize_deflicker(deflicker_frames_folder, deflicker_frames_output_folder, ddf_strength, over_strength, norm_strength):
    if norm_strength <= 0: # Condición 1: Norm_strength debe ser mayor a 0
        return
    
    # Si ddf_strength y/o over_strength son mayores a 0, utilizar deflicker_frames_output_folder en lugar de deflicker_frames_folder
    if ddf_strength > 0 or over_strength > 0:
        deflicker_frames_folder = deflicker_frames_output_folder
    
    # Crear la carpeta GenOverNorm si no existe
    if not os.path.exists("normtemp"):
        os.makedirs("normtemp")
        
    if not os.path.exists(deflicker_frames_output_folder):
        os.makedirs(deflicker_frames_output_folder)

    img_list = os.listdir(deflicker_frames_folder)
    img_list.sort()
        
    for i in tqdm(range(len(img_list)-1), desc="Normalizing frames"):
        normalize_frame(
            os.path.join(deflicker_frames_folder, img_list[i]),
            os.path.join(deflicker_frames_folder, img_list[i+1]),
            img_list[i+1],
        )
    
    # Copiar la primera imagen en la carpeta GenOverNorm para mantener la secuencia completa
    img0 = cv2.imread(os.path.join(deflicker_frames_folder, img_list[0]))
    cv2.imwrite(os.path.join("normtemp", img_list[0]), img0)
    
    # Definimos la ruta de la carpeta "overtemp"
    ruta_overtemp = "normtemp"
    
    # Movemos todos los archivos de la carpeta "overtemp" a la carpeta "frames_output_folder"
    for archivo in tqdm(os.listdir(ruta_overtemp), desc="Moving Norm Images"):
        origen = os.path.join(ruta_overtemp, archivo)
        destino = os.path.join(deflicker_frames_output_folder, archivo)
        shutil.move(origen, destino)
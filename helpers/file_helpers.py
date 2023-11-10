import os
import shutil
from PIL import Image
from tqdm import tqdm

def copy_images(
    reference_frames_folder,
    source_folder, 
    frames_limit=0,
    frame_indicies=None,
):
    # Copy all the images from the folder reference_frames_folder to the folder Source
    count = 0
    
    files = os.listdir(reference_frames_folder)
    sorted_files = sorted(files)

    if frame_indicies is not None:
        sorted_files = [sorted_files[i] for i in frame_indicies]
    
    for i, file in tqdm(enumerate(sorted_files), total=len(sorted_files), desc="Copying Source Images"):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img = Image.open(os.path.join(reference_frames_folder, file))
            rgb_img = img.convert('RGB')
            rgb_img.save(os.path.join(source_folder, "{:05d}.jpeg".format(i+1)), "jpeg", quality=100)
            count += 1
            if frames_limit > 0 and count >= frames_limit:
                break

    print(f"Total frames copied: {count}")


def copy_rename_images(
    frames_folder,
    destination_folder,
    filename_format = "{:05d}",
):
    # Get a list of file names in the generated_frames_folder folder, sorted
    file_list = sorted(os.listdir(frames_folder))
    # Rename all of the files
    for i, file_name in tqdm(enumerate(file_list), total=len(file_list), desc=f"Copy-Renaming Source Images with {filename_format} format"):
        old_path = os.path.join(frames_folder, file_name) # old path of the file
        new_file_name = filename_format.format(i+1) # new file name with format %05d
        new_path = os.path.join(destination_folder, new_file_name + os.path.splitext(file_name)[1]) # nueva ruta del archivo
        try:
            # copy the file over
            shutil.copy(old_path, new_path)
        except FileExistsError:
            print(f"File {new_file_name} already exists. Rename skipped.")


def remove_first_file(
    folder,
):
    maskd_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.startswith('')]
    if maskd_files:
        maskd_file = os.path.join(folder, sorted(maskd_files)[0])
        os.remove(maskd_file)
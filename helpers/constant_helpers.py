import os
import shutil

MAIN_EXTENSION_FOLDER = "./extensions/Abysz-LAB-Ext"
RUN_FOLDER = f'{MAIN_EXTENSION_FOLDER}/Run'

FULL_SCRIPT_FOLDER = f'{RUN_FOLDER}/Source'
MASK_DIRECTORY_FOLDER = f'{RUN_FOLDER}/MaskD'
MASK_TEMP_SOURCE_FOLDER = f'{RUN_FOLDER}/MaskS'
OUTPUT_FOLDER = f'{RUN_FOLDER}/Output'
GEN_FOLDER = f'{RUN_FOLDER}/Gen'
OVERLAY_FOLDER = f'{RUN_FOLDER}/Overlay'


def init_deflicker_folders():
    overlay_folder = init_folder(OVERLAY_FOLDER)
    return (overlay_folder,overlay_folder)


def init_project_folders(
    maskD_override = None,
    maskS_override = None,
    output_override = None,
    source_override = None,
    gen_override = None,
):
    maskD = init_folder(maskD_override or MASK_DIRECTORY_FOLDER)
    maskS = init_folder(maskS_override or MASK_TEMP_SOURCE_FOLDER)
    output = init_folder(output_override or OUTPUT_FOLDER)
    source = init_folder(source_override or FULL_SCRIPT_FOLDER)
    gen = init_folder(gen_override or GEN_FOLDER)

    return maskD, maskS, output, source, gen


def init_folder(folder_location):
    full_path = os.path.join(os.getcwd(), folder_location)
    if os.path.exists(folder_location): # verify if the folder exists
        shutil.rmtree(folder_location) # delete the folder and its content
    os.makedirs(folder_location, exist_ok=True)
    print(f"Folder {folder_location} created.")
    return full_path


def destroy_project_folders():
    shutil.rmtree(RUN_FOLDER)

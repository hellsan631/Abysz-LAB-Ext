import os
import shutil

full_script_folder = "./extensions/Abysz-LAB-Ext/scripts/Run/Source"
mask_directory_folder = './extensions/Abysz-LAB-Ext/scripts/Run/MaskD'
mask_temp_source_folder = './extensions/Abysz-LAB-Ext/scripts/Run/MaskS'
output_folder = './extensions/Abysz-LAB-Ext/scripts/Run/Output'
gen_folder = './extensions/Abysz-LAB-Ext/scripts/Run/Gen'


def init_project_folders(
    maskD_override = None,
    maskS_override = None,
    output_override = None,
    source_override = None,
    gen_override = None,
):
    maskD = init_folder(maskD_override or mask_directory_folder)
    maskS = init_folder(maskS_override or mask_temp_source_folder)
    output = init_folder(output_override or output_folder)
    source = init_folder(source_override or full_script_folder)
    gen = init_folder(gen_override or gen_folder)

    return maskD, maskS, output, source, gen


def init_folder(folder_location):
    full_path = os.path.join(os.getcwd(), folder_location)
    if os.path.exists(folder_location): # verify if the folder exists
        shutil.rmtree(folder_location) # delete the folder and its content
    os.makedirs(folder_location, exist_ok=True)
    print(f"Folder {folder_location} created.")
    return full_path

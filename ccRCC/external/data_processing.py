import os
import shutil


def cover_files(source_dir, target_ir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, target_ir)

def remove_file(old_path, new_path):
    shutil.copy(old_path, new_path)

def copy_file(old_path, file, new_path):
    new_path_make = os.path.join(new_path, file)
    mkdir(new_path_make)  # person
    person_path = os.path.join(old_path, file)
    for image in os.listdir(person_path):
        image_path = os.path.join(person_path, image)
        remove_file(image_path, new_path_make)

def delete_file(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

def make_person_file(path, save_path):
    mkdir(save_path)
    ensure_dir_exists(path)
    path_dir = os.listdir(path)  # Take the original path of the picture
    name_list = []
    for name in path_dir:  # get the namelist
        name_list.append(name[:12])  # start from index with 0
    name_list = set(name_list)
    for x in name_list:
        person_file = save_path + '/' + x
        mkdir(person_file)
        for file in path_dir:
            if file[:12] == x:
                old_path = path + file
                new_path = person_file + '/'
                remove_file(old_path, new_path)

def get_id_label(file):
    id_label = []
    id = 0
    label = 0
    with open(file, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            count = 0
            if not lines:
                break
                pass
            for i in lines.split():
                count = count + 1
                if count == 2:
                    id = i
                if count == 9:
                    label = i
                    id_label.append((id, label))
                    break

    return id_label



def class_file(id_label, person_path, test_path_0, test_path_1):
    path_dir = os.listdir(person_path)  # list
    for file in path_dir:
        for label in id_label:
            if file == label[0]:
                if (label[1] == "G1") or (label[1] == "G2"):
                    print("0", label[1])
                    copy_file(person_path, file, test_path_0)
                elif (label[1] == "G3") or (label[1] == "G4"):
                    print("1", label[1])
                    copy_file(person_path, file, test_path_1)


#  spilt train and val data
if __name__ == '__main__':
    # spilt classification
    person_path = "./TCGA_KIRC/"
    file = "./TCGA_KIRC_tumer/nationwidechildrens.org_clinical_patient_kirc.txt"
    test_path_0 = "./TCGA_KIRC_tumer/0/"
    test_path_1 = "./TCGA_KIRC_tumer/1/"
    # get the id and label
    id_label = get_id_label(file)
    class_file(id_label, person_path, test_path_0, test_path_1)

    # delete_file(test_path_0)
    # delete_file(test_path_1)




































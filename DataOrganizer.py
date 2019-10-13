import os,csv,shutil

class DataOrganizer:

    def __init__(self,data_set_path):
        self.dataset_file_path = data_set_path
        self.image_files_path='train'
        self.organized_data_folder='organized-data'
        self.image_label_dictionary = {}
        self.total_class_counts = {}
        self.class_train_and_test_counts = {}

    def create_folder_paths(self):
        folder_paths_to_create = ['/test/','/train/']
        for key in self.image_label_dictionary:
            for path in folder_paths_to_create:
                try:
                    os.mkdir(os.getcwd()+self.dataset_file_path+
                             self.organized_data_folder+path+self.image_label_dictionary[key])
                except:
                    pass

    def get_absolute_file_path(self,set_file_path):
        return os.listdir(os.getcwd()+self.dataset_file_path+set_file_path)

    def get_file_id(self,file_name,file_type):
        return file_name.split(file_type)[0]

    def determine_class_counts(self,dataset):
        for file in self.get_absolute_file_path(dataset):
            file_id = self.get_file_id(file,'.jpg')
            print('copying file id: '+file_id)
            dog_breed = self.image_label_dictionary[file_id]
            if self.total_class_counts.get(dog_breed) is None:
                self.total_class_counts[dog_breed] = 1
            else:
                self.total_class_counts[dog_breed] += 1
        for label in self.total_class_counts:
            total_images = self.total_class_counts[label]
            number_of_training_images = round(0.75*total_images)
            number_of_testing_images = total_images - number_of_training_images
            self.class_train_and_test_counts[label] = {
                "train":number_of_training_images,
                "test":number_of_testing_images
            }

    def set_image_label_dictionary(self):
        with open('dog-breed-identification\labels.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.image_label_dictionary[row[0]] = row[1]
            del self.image_label_dictionary['id']

    def determine_train_or_test_set(self,label):
        if self.class_train_and_test_counts[label]["test"] > 0:
            self.class_train_and_test_counts[label]["test"] -= 1
            return "train"
        else:
            return "test"

    def create_image_folders(self):
        for file in self.get_absolute_file_path(self.image_files_path):
            file_id = self.get_file_id(file,'.jpg')
            print('copying file id: '+file_id)
            label = self.image_label_dictionary[file_id]
            dataset = self.determine_train_or_test_set(label)
            shutil.copy(os.getcwd()+self.dataset_file_path+self.image_files_path+"/"+file,os.getcwd()+
                        self.dataset_file_path+self.organized_data_folder+'/'+dataset+'/'+label)


data_organizer = DataOrganizer('\dog-breed-identification\\')
data_organizer.set_image_label_dictionary()
data_organizer.create_folder_paths()
data_organizer.determine_class_counts(data_organizer.image_files_path)
print(data_organizer.total_class_counts)
data_organizer.create_image_folders()
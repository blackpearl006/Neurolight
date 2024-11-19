import os
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset

class Fieldmapdata(Dataset):
    def __init__(self, root_dir, label_dir, task='classification'):
        self.labels_df = self.load_labels(label_dir)
        self.samples = self.make_dataset(root_dir, task=task)

    def __len__(self):
        return len(self.samples)
    
    def __getposweight__(self):
        sexs = []
        for path, sex in self.samples:
            sexs.append(sex)
        return sum(sexs)/(len(sexs)-sum(sexs))

    def __getitem__(self, idx):
        img_path, label, dataset, id_ = self.samples[idx]
        nifti_data = nib.load(img_path)
        data = nifti_data.get_fdata()
        image_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        label_tensor = (label_tensor, dataset, id_)
        return image_tensor, label_tensor
       
    def make_dataset(self, root_dir, task = 'regression'):
        samples = []
        labels_df = self.labels_df
        for root, _, fnames in os.walk(root_dir):
            for fname in fnames:
                if fname.endswith(".nii.gz") or fname.endswith(".nii"):
                    path = os.path.join(root, fname)
                    id_ = self.extract_id_from_filename(fname)
                    try:
                        if task == 'regression': 
                            label = labels_df[labels_df['Subject'] == id_]['Age'].iloc[0] 
                        else: 
                            label = labels_df[labels_df['Subject'] == id_]['Gender'].iloc[0]
                        samples.append((path, label, 'HCP', id_))
                    except:
                        continue
        return samples
    
    def extract_id_from_filename(self, fname):
        fname = fname.replace("sub-","")
        if fname.endswith("_ad.nii.gz"):
            id_ = fname.replace("_ad.nii.gz", "")
        elif fname.endswith("_rd.nii.gz"):
            id_ = fname.replace("_rd.nii.gz", "")
        elif fname.endswith("_adc.nii.gz"):
            id_ = fname.replace("_adc.nii.gz", "")
        elif fname.endswith("_fa.nii.gz"):
            id_ = fname.replace("_fa.nii.gz", "")
        elif fname.endswith(".nii.gz"):
            id_ = fname.replace(".nii.gz", "")
        return id_

    def load_labels(self, label_path):
        df = pd.read_csv(label_path)
        df_filtered = df[['Subject', 'Gender', 'Age']].copy()
        df_filtered['Gender'] = df_filtered['Gender'].map({'M': 0, 'F': 1}).astype(int)
        df_filtered['Age'] = df_filtered['Age'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) // 2 if '-' in x else int(x[:-1])).astype(float)
        df_filtered['Subject'] = df_filtered['Subject'].astype(str)
        return df_filtered
    
class Fieldmapdata_4updated(Dataset):
    def __init__(self, root_dir, label_dir, task='classification'):
        self.labels_df = self.load_labels(label_dir)
        self.samples = self.make_dataset(root_dir, task=task)

    def __len__(self):
        return len(self.samples)
    
    def __getposweight__(self):
        sexs = []
        for paths, sex in self.samples:
            sexs.append(sex)
        return sum(sexs) / (len(sexs) - sum(sexs))

    def __getitem__(self, idx):
        img_paths, label, dataset, id_ = self.samples[idx]
        nifti_tensors = []
        for img_path in img_paths:
            nifti_data = nib.load(img_path)
            data = nifti_data.get_fdata()
            nifti_tensors.append(torch.tensor(data, dtype=torch.float32))
        
        image_tensor = torch.stack(nifti_tensors, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        label_tensor = (label_tensor, dataset, id_)
        return image_tensor, label_tensor
       
    def make_dataset(self, root_dir, task='regression'):
        samples = []
        labels_df = self.labels_df

        # Define subdirectories corresponding to the four fieldmaps
        subdirs = ['AD', 'RD', 'ADC', 'FA']

        # Collect subject IDs and their corresponding fieldmap paths
        subject_files = {}
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            for fname in os.listdir(subdir_path):
                if fname.endswith(".nii.gz") or fname.endswith(".nii"):
                    id_ = self.extract_id_from_filename(fname)
                    path = os.path.join(subdir_path, fname)
                    if id_ not in subject_files:
                        subject_files[id_] = {}
                    subject_files[id_][subdir] = path

        # Ensure each subject has all four fieldmaps
        for id_, paths_dict in subject_files.items():
            if all(subdir in paths_dict for subdir in subdirs):  # Check if all four fieldmaps are present
                try:
                    paths = [paths_dict[subdir] for subdir in subdirs]  # Ordered list of paths
                    if task == 'regression': 
                        label = labels_df[labels_df['Subject'] == id_]['Age'].iloc[0]
                    else:
                        label = labels_df[labels_df['Subject'] == id_]['Gender'].iloc[0]
                    samples.append((paths, label, 'HCP', id_))
                except Exception:
                    continue
        return samples
    
    def extract_id_from_filename(self, fname):
        fname = fname.replace("sub-","")
        if fname.endswith("_ad.nii.gz"):
            id_ = fname.replace("_ad.nii.gz", "")
        elif fname.endswith("_rd.nii.gz"):
            id_ = fname.replace("_rd.nii.gz", "")
        elif fname.endswith("_adc.nii.gz"):
            id_ = fname.replace("_adc.nii.gz", "")
        elif fname.endswith("_fa.nii.gz"):
            id_ = fname.replace("_fa.nii.gz", "")
        elif fname.endswith(".nii.gz"):
            id_ = fname.replace(".nii.gz", "")
        return id_

    def load_labels(self, label_path):
        df = pd.read_csv(label_path)
        df_filtered = df[['Subject', 'Gender', 'Age']].copy()
        df_filtered['Gender'] = df_filtered['Gender'].map({'M': 0, 'F': 1}).astype(int)
        df_filtered['Age'] = df_filtered['Age'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) // 2 if '-' in x else int(x[:-1])).astype(float)
        df_filtered['Subject'] = df_filtered['Subject'].astype(str)
        return df_filtered

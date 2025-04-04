import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import re
from dataclasses import dataclass


class RadarDataset(Dataset):
    def __init__(self, i_data_dir, q_data_dir, tfm_bp_dir, imf_range=(5, 8), transform=None):
        self.i_data_dir = i_data_dir
        self.q_data_dir = q_data_dir
        self.tfm_bp_dir = tfm_bp_dir
        self.imf_range = imf_range  
        self.transform = transform
        
        self.i_files = sorted(os.listdir(i_data_dir))
        self.q_files = sorted(os.listdir(q_data_dir))
        self.tfm_bp_files = sorted(os.listdir(tfm_bp_dir))
        
    def __len__(self):
        return len(self.i_files)
    
    def __getitem__(self, idx):
        
        i_file = os.path.join(self.i_data_dir, self.i_files[idx])
        q_file = os.path.join(self.q_data_dir, self.q_files[idx])
        tfm_bp_file = os.path.join(self.tfm_bp_dir, self.tfm_bp_files[idx])

       
        i_base = os.path.splitext(self.i_files[idx])[0].rsplit('_radar_i_', 1)[0]
        q_base = os.path.splitext(self.q_files[idx])[0].rsplit('_radar_q_', 1)[0]
        tfm_base = os.path.splitext(self.tfm_bp_files[idx])[0].rsplit('_tfm_bp_', 1)[0]

        i_num = os.path.splitext(self.i_files[idx])[0].split('_')[-1]
        q_num = os.path.splitext(self.q_files[idx])[0].split('_')[-1] 
        tfm_num = os.path.splitext(self.tfm_bp_files[idx])[0].split('_')[-1]

  
        i_data = loadmat(i_file)['real_segment']  
        q_data = loadmat(q_file)['imag_segment']

        tfm_bp_data = loadmat(tfm_bp_file)['tfm_bp_segment']
        
        
        i_data_sub = i_data[:, self.imf_range[0]-1:self.imf_range[1]]  # Python index starts at 0
        q_data_sub = q_data[:, self.imf_range[0]-1:self.imf_range[1]]
           
        combined_data = np.concatenate([i_data_sub, q_data_sub], axis=1)

        combined_data = combined_data.T

        iq_data = i_data_sub + 1j*q_data_sub
        max_amp = np.max(np.abs(iq_data))
        combined_data = combined_data / max_amp
        
       
        combined_data = torch.tensor(combined_data, dtype=torch.float32)
           
        tfm_bp_data = torch.tensor(tfm_bp_data, dtype=torch.float32)
        
        if self.transform:
            combined_data = self.transform(combined_data)
            tfm_bp_data = self.transform(tfm_bp_data)
        
        return combined_data, tfm_bp_data

def create_data_loaders(i_data_dir, q_data_dir, tfm_bp_dir, batch_size=256, imf_range=(5, 8), shuffle_=True, num_workers_=4):
    dataset = RadarDataset(i_data_dir, q_data_dir, tfm_bp_dir, imf_range)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_, num_workers=num_workers_)  
    return data_loader

class SingleFileDataset(Dataset):
    def __init__(self, i_data_path, q_data_path, tfm_bp_path, imf_range=(5, 8), flag_info=False):
        self.i_data = loadmat(i_data_path)['real_segment']
        self.q_data = loadmat(q_data_path)['imag_segment']
        self.tfm_bp = loadmat(tfm_bp_path)['tfm_bp_segment_fit']
        self.imf_range = imf_range

        tfm_bp_filename = os.path.basename(tfm_bp_path)
        self.tfm_bp_path = tfm_bp_path

        self.flag_info = flag_info

        pattern = r'GDN(\d{4})_(\w+)_tfm_bp_(\d+)\.mat'
        match = re.match(pattern, tfm_bp_filename)
        
        if match:
            self.patient_num = int(match.group(1))  
            self.scene = match.group(2)             
            self.seq_num = int(match.group(3))      
        

    def __len__(self):
        return len(self.i_data)

    def __getitem__(self, idx):
        i_sample = self.i_data[:, self.imf_range[0]-1:self.imf_range[1]]
        q_sample = self.q_data[:, self.imf_range[0]-1:self.imf_range[1]]
              
        combined_data = np.concatenate([i_sample, q_sample], axis=1)

        combined_data = combined_data.T

        iq_data = i_sample + 1j*q_sample
        max_amp = np.max(np.abs(iq_data))
        combined_data = combined_data / max_amp

        combined_data = torch.tensor(combined_data, dtype=torch.float32)
        
        a_norm = 180 
        b_norm = 40
        tfm_bp_data_norm = (self.tfm_bp - b_norm) / (a_norm - b_norm) * 2 - 1  
        tfm_bp_data = torch.tensor(tfm_bp_data_norm, dtype=torch.float32)
        
        
        if not self.flag_info:
            return combined_data, tfm_bp_data
        else:
            
            return combined_data, tfm_bp_data, self.tfm_bp_path


def create_specific_data_loader(i_data_dir, q_data_dir, tfm_bp_dir, patient_num, scene, seq_num, batch_size=1, imf_range=(5, 8)):
    i_data_path = os.path.join(i_data_dir, f"GDN{patient_num:04d}_{scene}_radar_i_{seq_num}.mat")
    q_data_path = os.path.join(q_data_dir, f"GDN{patient_num:04d}_{scene}_radar_q_{seq_num}.mat")
    tfm_bp_path = os.path.join(tfm_bp_dir, f"GDN{patient_num:04d}_{scene}_tfm_bp_{seq_num}.mat")
    
    dataset = SingleFileDataset(i_data_path, q_data_path, tfm_bp_path, imf_range)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader


class FolderRadarDataset(Dataset):
    def __init__(self, i_data_dir, q_data_dir, tfm_bp_dir, 
                 imf_range=(5, 8), transform=None, return_paths=False):
       
        self.i_data_dir = i_data_dir
        self.q_data_dir = q_data_dir
        self.tfm_bp_dir = tfm_bp_dir
        self.imf_range = imf_range
        self.transform = transform
        self.return_paths = return_paths
        
        i_files = sorted([f for f in os.listdir(i_data_dir) if f.endswith('.mat')])
        q_files = sorted([f for f in os.listdir(q_data_dir) if f.endswith('.mat')])
        tfm_bp_files = sorted([f for f in os.listdir(tfm_bp_dir) if f.endswith('.mat')])
        
        self.i_paths = [os.path.join(i_data_dir, f) for f in i_files]
        self.q_paths = [os.path.join(q_data_dir, f) for f in q_files]
        self.tfm_bp_paths = [os.path.join(tfm_bp_dir, f) for f in tfm_bp_files]
        
       

    def __len__(self):
        return len(self.i_paths)
    
    def __getitem__(self, idx):
       
        i_file = self.i_paths[idx]
        q_file = self.q_paths[idx]
        tfm_bp_file = self.tfm_bp_paths[idx]

        i_name = os.path.splitext(os.path.basename(i_file))[0]
        q_name = os.path.splitext(os.path.basename(q_file))[0]
        tfm_bp_name = os.path.splitext(os.path.basename(tfm_bp_file))[0]

        i_parts = i_name.split("_")
        q_parts = q_name.split("_")
        tfm_bp_parts = tfm_bp_name.split("_")
        
        i_base = "_".join(i_parts[:2])    
        q_base = "_".join(q_parts[:2])    
        tfm_bp_base = "_".join(tfm_bp_parts[:2])
        
        i_num = i_parts[-1]
        q_num = q_parts[-1]
        tfm_bp_num = tfm_bp_parts[-1]

       
        i_data = loadmat(i_file)['real_segment']  
        q_data = loadmat(q_file)['imag_segment']  
        
        tfm_bp_data = loadmat(tfm_bp_file)['tfm_bp_segment']  

        i_data_sub = i_data[:, self.imf_range[0]-1:self.imf_range[1]]
        q_data_sub = q_data[:, self.imf_range[0]-1:self.imf_range[1]]

        combined_data = np.concatenate([i_data_sub, q_data_sub], axis=1)
        combined_data = combined_data.T  
        
        iq_data = i_data_sub + 1j * q_data_sub
        max_amp = np.max(np.abs(iq_data))
        combined_data = combined_data / max_amp

        combined_data = torch.tensor(combined_data, dtype=torch.float32)

        tfm_bp_data = torch.tensor(tfm_bp_data, dtype=torch.float32)

        if self.transform is not None:
            combined_data = self.transform(combined_data)
            tfm_bp_data = self.transform(tfm_bp_data)

        
        if self.return_paths:
            return combined_data, tfm_bp_data, i_file, q_file, tfm_bp_file
        else:
            return combined_data, tfm_bp_data


def create_specific_folder_data_loader(i_data_dir, q_data_dir, tfm_bp_dir, 
                        batch_size=256, imf_range=(5, 8), 
                        shuffle_=True, num_workers_=4, 
                        return_paths=False):
   
    dataset = FolderRadarDataset(
        i_data_dir=i_data_dir,
        q_data_dir=q_data_dir,
        tfm_bp_dir=tfm_bp_dir,
        imf_range=imf_range,
        transform=None,
        return_paths=return_paths
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_, 
        num_workers=num_workers_
    )
    return data_loader


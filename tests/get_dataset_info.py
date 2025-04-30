import h5py

def print_hdf5_structure(filename):
    """打印HDF5文件的基本结构"""
    with h5py.File(filename, 'r') as f:
        print(f"文件: {filename}")
        
        # 显示顶级键
        print("顶级组:")
        for key in f.keys():
            print(f"- {key}")
            
            # 如果是组，显示其内部结构
            if isinstance(f[key], h5py.Group):
                for subkey in f[key].keys():
                    print(f"  - {key}/{subkey}")
                    
                    # 如果是数据集，显示形状和类型
                    if isinstance(f[key][subkey], h5py.Dataset):
                        dataset = f[key][subkey]
                        print(f"    形状: {dataset.shape}, 类型: {dataset.dtype}")
                    elif isinstance(f[key][subkey], h5py.Group):
                        for subsubkey in f[key][subkey].keys():
                            print(f"    - {key}/{subkey}/{subsubkey}")
                            if isinstance(f[key][subkey][subsubkey], h5py.Dataset):
                                dataset = f[key][subkey][subsubkey]
                                print(f"        形状: {dataset.shape}, 类型: {dataset.dtype}")
                            elif isinstance(f[key][subkey][subsubkey], h5py.Group):
                                for subsubsubkey in f[key][subkey][subsubkey].keys():
                                    print(f"        - {key}/{subkey}/{subsubkey}/{subsubsubkey}")
                                    if isinstance(f[key][subkey][subsubkey][subsubsubkey], h5py.Dataset):
                                        dataset = f[key][subkey][subsubkey][subsubsubkey]
                                        print(f"            形状: {dataset.shape}, 类型: {dataset.dtype}")
                break

# 使用方法
print_hdf5_structure("pusht_image.hdf5")

import numpy as np

def load_data(landmark_file, target_file):
    landmark=np.loadtxt(landmark_file, delimiter='\t',dtype=str)[:,4:].astype(np.float32).T
    target=np.loadtxt(target_file,delimiter='\t',dtype=str)[:,4:].astype(np.float32).T

    full_data=np.concatenate([landmark, target], axis=1)

    return full_data


def main():
    full_data=load_data("/home/ibab/Deep_Learning/DL_Lab10/Landmark_genes.txt",
                        "/home/ibab/Deep_Learning/DL_Lab10/Target_genes.txt")
    print("Final shape:", full_data.shape)

if __name__ == "__main__":
    main()


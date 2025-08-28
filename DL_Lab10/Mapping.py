def get_ids():
    main_file="/home/ibab/Deep_Learning/DL_Lab10/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt"
    landmark_file="/home/ibab/Deep_Learning/DL_Lab10/map_lm.txt"
    target_file = "/home/ibab/Deep_Learning/DL_Lab10/map_tg.txt"

    landmark_ids=set(line.strip().split()[1] for line in open(landmark_file))
    target_ids=set(line.strip().split()[1] for line in open(target_file))

    return landmark_ids, target_ids, main_file

def base_gene_id(gene_id):
    return gene_id.split('.')[0]

def filter_genes(ids,main_file,output_file):
    with open(main_file) as fin, open(output_file,"w") as fout:
        for line in fin:
            gene_id=line.split()[0]
            if base_gene_id(gene_id) in ids:
                fout.write(line)

def main():
    landmark_ids, target_ids, main_file = get_ids()
    filter_genes(landmark_ids,main_file,"Landmark_genes.txt")
    filter_genes(target_ids, main_file, "Target_genes.txt")

    print(f"Landmark: {sum(1 for _ in open('Landmark_genes.txt'))},"
          f"Target: {sum(1 for _ in open('Target_genes.txt'))}")

    print("Landmark:", (sum(1 for _ in open("Landmark_genes.txt")),
                        len(open("Landmark_genes.txt").readline().split())))

    print("Target:", (sum(1 for _ in open("Target_genes.txt")),
                      len(open("Target_genes.txt").readline().split())))

if __name__ == "__main__":
    main()



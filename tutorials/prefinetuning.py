from methylbert.data import finetune_data_generate as fdg

f_bam_file_list = "bam_list.txt"
f_dmr = "dmrs_ctype.csv"
f_ref = "/home/wuyuwei/nobackup/other/hg38/hg38.fa"
out_dir = "tmp/"

fdg.finetune_data_generate(
    sc_dataset = f_bam_file_list,
    f_dmr = f_dmr,
    f_ref = f_ref,
    output_dir=out_dir,
    split_ratio = 0.8, # Split ratio to make training and validation data
    n_mers=3, # 3-mer DNA sequences 
    n_cores=20
)

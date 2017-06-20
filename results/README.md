**all_results_compact_form.csv** — all results in compact form

Columns

*site* — data site, either BNU_1, HNU_1 or IPCAS_1

*tracktography* — tractography method (probabilistic or deterministic) 

*rec_model* — reconstruction model (CSA, CSD, DTI)

*parcellation* — parcellation. Either Desikan-Killiany (aparc68), Destriex (aparc150) or Lausanne2008(scaleX, where X is number of ROIs per hemisphere, more info on https://github.com/mattcieslak/easy_lausanne)

*norm* — connectome normalization

*features* — connectome features

*PACC_mean{std}* — pairwise accuracy mean(std) on evaluation splits

*gender_eval_accuracy_mean{std}* — gender classification accuracy mean(std) on evaluation splits

*icc_mean{std, max, median}* — ICC mean, std, max, median by features  

# Action-gradient engineering repairs

64766: TC1N05 V100 allocation，FAILED1:0/14s。checkpoint、source/helper、model structure与dataset metadata mapping检查通过，首个target初始化后、任何candidate score/gradient计算前，调用np.default_rng出错（应为np.random.default_rng）。完整run.log与原始作业保留；这是NumPy namespace接线错误，未产生scientific no-go。root仅将传入_candidate_pool的namespace改为np.random，seed与shape不变，不改protocol。

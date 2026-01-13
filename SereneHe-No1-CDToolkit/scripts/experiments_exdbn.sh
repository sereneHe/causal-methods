#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export  KMP_DUPLICATE_LIB_OK=TRUE

CMD="python3 run_experiments.py --multirun --config-name=config-cluster experiment='EXDBN2'"

vars="7, 10, 15, 20, 25"

# ER-3-1

${CMD} problem="dynamic" problem.generator="notears" problem.normalize="false"  problem.graph_type_intra="er" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.number_of_variables="${vars}" problem.number_of_samples="50, 100, 250, 500, 1000" solver="milp" solver.target_mip_gap="0.00001" solver.time_limit="7200"  problem.p="1" solver.lambda1="2,1,0.1" solver.lambda2="0.1,0.5, 0.01" solver.a_reg_type="l1,l2" problem.seed="range(0,10)"
#${CMD} problem="dynamic" problem.graph_type_intra="er" problem.normalize="false" problem.generator="notears" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.number_of_variables="${vars}" problem.number_of_samples="50, 100, 250, 500, 1000" solver="lingam,dynotears" problem.p="1" solver.lambda1="0.03" problem.seed="range(0,10)"

# SF-3-1
#${CMD} problem="dynamic" problem.graph_type_intra="sf" problem.generator="notears" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.number_of_variables="${vars}" problem.number_of_samples="50, 100, 250, 500, 1000" solver="lingam,dynotears" problem.p="1" solver.lambda1="0.03" problem.seed="range(0,10)"
#${CMD} experiment="EXDBN" problem="dynamic" problem.generator="notears" problem.variant="sf" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.number_of_variables="${vars}" problem.number_of_samples="50, 100, 250, 500, 1000" solver="milp"  solver.time_limit="7200"  problem.p="1" solver.lambda1="2,1" solver.lambda2="0.1,0.5" solver.a_reg_type="l1,l2" problem.seed="range(0,10)"

# ER-2-1-1

#${CMD} problem="dynamic" problem.generator="notears" problem.graph_type_intra="er" problem.normalize="false" problem.intra_edge_ratio=2 problem.w_min_inter=0.3 problem.w_max_inter=0.5  problem.number_of_variables="5, 10, 15, 20, 25" problem.number_of_samples="50, 100, 250, 500, 1000" solver="milp"  problem.p="2" problem.w_decay=3 solver.lambda1="3,1, 0.3, 0.1,0.01" solver.time_limit="7200"  solver.lambda2="0.1,0.01" solver.a_reg_type="l1,l2" problem.seed="range(0,10)"
#${CMD} problem="dynamic" problem.graph_type_intra="er" problem.generator="notears" problem.w_min_inter=0.3 problem.w_max_inter=0.5  problem.intra_edge_ratio=2 problem.number_of_variables="5,10,15,20,25" problem.number_of_samples="50, 100, 250, 500, 1000" solver="dynotears,lingam" problem.p="2" problem.normalize="false" problem.w_decay=3 solver.lambda1="0.03" problem.seed="range(0,10)"


# ER-3-1-var
#${CMD} problem="dynamic" problem.generator="notears" problem.normalize="false"  problem.graph_type_intra="er" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.noise_scale=0.8 problem.noise_scale_variance=0.4  problem.number_of_variables="${vars}" problem.number_of_samples="50, 100, 250, 500, 1000" solver="milp" solver.target_mip_gap="0.00001" solver.time_limit="7200"  problem.p="1" solver.lambda1="2,1,0.1" solver.lambda2="0.1,0.5, 0.01" solver.a_reg_type="l1,l2" problem.seed="range(0,10)"
#${CMD} problem="dynamic" problem.graph_type_intra="er" problem.normalize="false" problem.noise_scale=0.8 problem.noise_scale_variance=0.4  problem.generator="notears" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.number_of_variables="${vars}" problem.number_of_samples="50, 100, 250, 500, 1000" solver="lingam,dynotears" problem.p="1" solver.lambda1="0.03" problem.seed="range(0,10)"


# convergence
#${CMD} experiment="EXDBN" problem="dynamic"  problem.normalize="false" problem.generator="notears" problem.variant="sf" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.number_of_variables="25" problem.number_of_samples="250" solver="milp"  solver.time_limit="60, 120, 300, 600, 1200, 1800, 2700, 3600, 5400, 7200"  problem.p="1" solver.lambda1="1" solver.lambda2="0.1" solver.loss_type="l2" solver.a_reg_type="l1" problem.seed="0"

# lazy constraint mode comparison SF-3-1

#${CMD} problem="dynamic" problem.generator="notears" problem.graph_type_intra="sf" problem.w_min_inter=0.2 problem.w_max_inter=0.4  problem.intra_edge_ratio=3 problem.number_of_variables="${vars}" problem.number_of_samples="50, 100, 250, 500, 1000" solver="milp"  solver.time_limit="7200" solver.callback_mode="shortest_cycle, first_cycle" problem.p="1" solver.lambda1="2,1" solver.lambda2="0.1,0.5" solver.a_reg_type="l1,l2" problem.seed="range(0,10)"


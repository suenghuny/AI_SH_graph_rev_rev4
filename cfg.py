import argparse
# vessl_on
# map_name1 = '6h_vs_8z'
# GNN = 'GAT'
def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
    parser.add_argument("--simtime_per_frame", type=int, default=2, help="framerate 관련")
    parser.add_argument("--decision_timestep", type=int, default=4, help="decision timestep 관련")
    parser.add_argument("--ciws_threshold", type=float, default=1, help="ciws threshold")
    parser.add_argument("--per_alpha", type=float, default=0.5, help="PER_alpha")
    parser.add_argument("--per_beta", type=float, default=0.6, help="PER_beta")
    parser.add_argument("--sigma_init", type=float, default=1, help="sigma_init")
    parser.add_argument("--n_step", type=int, default=5, help="n_step")
    parser.add_argument("--anneal_episode", type=int, default=1500, help="episode")
    parser.add_argument("--vdn", type=bool, default=True, help="vdn")
    parser.add_argument("--map_name", type=str, default='6h_vs_8z', help="map name")
    parser.add_argument("--GNN", type=str, default='GCRN', help="map name")
    parser.add_argument("--hidden_size_comm", type=int, default=56, help="GNN hidden layer")
    parser.add_argument("--hidden_size_enemy", type=int, default=64, help="GNN hidden layer")
    parser.add_argument("--hidden_size_meta_path", type=int, default=64, help="GNN hidden layer")
    parser.add_argument("--hidden_size_meta_path2", type=int, default=64, help="GNN hidden layer")
    parser.add_argument("--iqn_layers", type=str, default= '[128,64,48,39,16]', help="layer 구조")
    parser.add_argument("--ppo_layers", type=str, default='[128,64,48,39,32]', help="layer 구조")
    parser.add_argument("--ship_layers", type=str, default='[126,108,64]', help="layer 구조")
    parser.add_argument("--missile_layers", type=str, default='[45,23]', help="layer 구조")
    parser.add_argument("--enemy_layers", type=str, default='[45,32]', help="layer 구조")
    parser.add_argument("--action_layers", type=str, default='[128,64]', help="layer 구조")
    parser.add_argument("--n_representation_ship", type=int, default=52, help="")
    parser.add_argument("--n_representation_missile", type=int, default=14, help="")
    parser.add_argument("--n_representation_enemy", type=int, default=28, help="")
    parser.add_argument("--n_representation_action", type=int, default=42, help="")
    parser.add_argument("--n_representation_action2", type=int, default=42, help="")
    parser.add_argument("--iqn_layer_size", type=int, default=64, help="")
    parser.add_argument("--iqn_N", type=int, default=48, help="")
    parser.add_argument("--n_cos", type=int, default=36, help="")
    parser.add_argument("--buffer_size", type=int, default=50000, help="")
    parser.add_argument("--batch_size", type=int, default=17, help="")
    parser.add_argument("--teleport_probability", type=float, default=1.0, help="teleport_probability")
    parser.add_argument("--gtn_beta", type=float, default=0.05, help="teleport_probability")
    parser.add_argument("--gamma", type=float, default=.99, help="discount ratio")
    parser.add_argument("--lr", type=float, default=0.88e-4, help="learning rate")
    parser.add_argument("--lr_min", type=float, default=0.5e-4, help="lr_min")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_multi_head", type=int, default=1, help="number of multi head")
    parser.add_argument("--num_episode", type=int, default=1000000, help="number of episode")
    parser.add_argument("--scheduler_step", type =int, default=1000, help= "scheduler step")
    parser.add_argument("--scheduler_ratio", type=float, default=0.5, help= "scheduler ratio")
    parser.add_argument("--train_start", type=int, default=1, help="number of train start")
    parser.add_argument("--epsilon_greedy", type=bool, default=True, help="epsilon_greedy")
    parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="epsilon")
    parser.add_argument("--anneal_step", type=int, default=50000, help="epsilon")
    parser.add_argument("--temperature", type=int, default=1, help="")
    parser.add_argument("--interval_min_blue", type=bool, default=True, help="interval_min_blue")
    parser.add_argument("--interval_constant_blue", type=float, default=1, help="interval_constant_blue")
    parser.add_argument("--action_history_step", type=int, default=4, help="action_history_step")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--grad_clip_step", type=int, default=50000, help="gradient clipping step")
    parser.add_argument("--grad_clip_reduce", type=float, default=1, help="reduced_gradient clipping")
    parser.add_argument("--test_epi", type=int, default=1800, help="interval_constant_blue")
    parser.add_argument("--scheduler", type=str, default='step', help="step 형태")
    parser.add_argument("--t_max", type=int, default=40000, help="interval_constant_blue")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lmbda")
    parser.add_argument("--eps_clip", type=float, default=0.18, help="clipping epsilon")
    parser.add_argument("--eps_clip_step", type=float, default=0.0005, help="clipping epsilon")
    parser.add_argument("--eps_clip_min", type=float, default=0.11, help="clipping epsilon")
    parser.add_argument("--K_epoch", type=int, default=2, help="K-epoch")
    parser.add_argument("--num_GT_layers", type=int, default=2, help="num GT layers")
    parser.add_argument("--channels", type=int, default=1, help="channels")
    parser.add_argument("--num_layers", type=int, default=2, help="num layers")
    parser.add_argument("--dropout", type=float, default=0, help="num layers")
    parser.add_argument("--embedding_train_stop", type=int, default=100, help="embedding_train_stop")
    parser.add_argument("--n_eval", type=int, default=20, help="number of evaluation")
    parser.add_argument("--with_noise", type=bool, default=False, help="")
    parser.add_argument("--temp_constant", type=float, default=1, help="")
    parser.add_argument("--init_constant", type=int, default=10000, help="")
    parser.add_argument("--cuda", type=str, default='cuda:0', help="")
    parser.add_argument("--num_action_history", type=int, default=10, help="")

    # 이녀석이 찐임
    parser.add_argument("--discr_n", type=int, default=12, help="")
    # 이녀석이 찐임
    parser.add_argument("--graph_distance", type=float, default=10, help="graph distance")
    # 이녀석이 찐임
    parser.add_argument("--bonus_reward", type=float, default=5, help="bonus_reward")
    # 이녀석이 찐임

    parser.add_argument("--optimizer", type=str, default='ADAM', help="optimizer")
    parser.add_argument("--entropy", type=bool, default=True, help="entropy")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--algorithm", type=str, default='ppo', help="algorithm")
    parser.add_argument("--leakyrelu", type=bool, default=False, help="attention mechanism leaky relu")
    parser.add_argument("--kl_target", type=float, default=0.005, help="kl target")
    parser.add_argument("--negativeslope", type=float, default=0.1, help="leaky relu negative slope")
    parser.add_argument("--n_data_parallelism", type=int, default=16, help="data parallelism")
    parser.add_argument("--k_hop", type=int, default=3, help="gnn k")
    parser.add_argument("--n_test", type=int, default=200, help="gnn k")
    parser.add_argument("--angle_random", type=bool, default=False, help="gnn k")
    parser.add_argument("--inception_angle", type=int, default=180, help="gnn k")

    return parser.parse_args()
# ISPRS and kidney
# class config:
#     ClusterNo = 3
#     batch_size = 1
#     learning_rate_decay_epochs = 3
#     data_root_path = r"./data/cells/"
#     img_size = 1024
#     epochs = 200
#     valid_freq = 1
#     log_path = './log/DGMN_lasso_ISPRS/'
#     valid_visualize_path = log_path + 'visual/'
#     train_print_step = 100
#     save_model_dir = log_path + 'weights/'
#     l2weight = 0.001
#     val_dice = True


# lasso-cells
# good lr-rate is 5e-5
class config:
    ClusterNo = 3
    batch_size = 1
    learning_rate_decay_epochs = 1
    data_root_path = r"E:\DATA\Cells\MICCAI/"
    img_size = 512
    epochs = 150
    valid_freq = 1
    log_path = './log/DGMM_lizard/'
    valid_visualize_path = log_path + 'visual/'
    train_print_step = 200
    save_model_dir = log_path + 'weights/'
    l2weight = 0.001
    val_dice = True
    load_weights = True


#supervised
# class config:
#     ClusterNo = 2
#     batch_size = 1
#     learning_rate_decay_epochs = 1
#     data_root_path = r"./data/cells/"
#     img_size = 896
#     epochs = 200
#     valid_freq = 1
#     log_path = './log/DCGNP_05/'
#     valid_visualize_path = log_path + 'visual/'
#     train_print_step = 100
#     save_model_dir = log_path + 'weights/'
#     l2weight = 0.001
#     val_dice = True

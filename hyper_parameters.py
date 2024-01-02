DNN_dict = {'hidden_dim1': 128, 'hidden_dim2': 64, 'batch_size': 256, 'lr': 0.2, 'epoch': 10}

DeepFM_dict = {'dnn_hidden_dim1': 16, 'dnn_hidden_dim2': 16, 'batch_size': 512, 'lr': 0.05, 'epoch': 10,
                    'hidden_dim1': 16, 'hidden_dim2': 16}

UniSRec_dict = {'moe_embedding_size': 16, 'moe_k': 4, 'tau': 0.9, 'lambda': 0.5, 'batch_size': 1024, 'lr': 0.001,
                     'epoch': 10}

MAMDR_dict = {'hidden_dim': 16, 'batch_size': 1024, 'lr': 0.1, 'epoch': 10, 'beta': 0.1, 'gamma': 0.2}

COR_dict = {'encoder_hidden_dim': 16, 'unobs_dim': 16, 'z1_hidden_dim': 16, 'z1_dim': 16, 'z2_hidden_dim': 16,
                 'z2_dim': 16, 'predict_hidden_dim': 16, 'beta': 0.1, 'batch_size': 1024, 'lr': 0.2, 'epoch': 10}

CausPref_dict = {'latent_dim': 8, 'rec_coe': 0.5, 'reg_alpha': 6.0, 'u2i_sparse_coe': 1e-06, 'bpr_coe': 1.0,
                      'item_sparse_coe': 1.0, 'batch_size': 1024, 'lr': 0.001, 'epoch': 10}

HIRL_RA_RF_dict = {'embedding_dim': 64, 'mask1_hidden_dim': 64, 'mask2_hidden_dim': 128,
                        'predictor_hidden_dim1': 64, 'predictor_hidden_dim2': 32, 'classifier1_hidden_dim': 32,
                        'classifier2_hidden_dim': 32, 'prob_hidden_dim': 32, 'batch_size': 1024, 'epoch': 10,
                        'invariant_lr': 0.2, 'invariant_loss_alpha1': 0.1, 'invariant_loss_alpha2': 0.1,
                        'invariant_loss_alpha3': 0.1, 'invariant_loss_alpha4': 0.1, 'classifier_lr': 0.001,
                        'classifier_loss_alpha': 1.0, 'environment_assign_lr': 0.001,
                        'environment_assign_loss_alpha': 1.0, 'environment_refine_noise_bound': 0.1}

HIRL_RA_heuAD_dict = {'embedding_dim': 32, 'mask1_hidden_dim': 32, 'mask2_hidden_dim': 64,
                           'predictor_hidden_dim1': 64, 'predictor_hidden_dim2': 32, 'classifier1_hidden_dim': 32,
                           'classifier2_hidden_dim': 32, 'prob_hidden_dim': 64, 'batch_size': 64, 'epoch': 10,
                           'invariant_lr': 0.05, 'invariant_loss_alpha1': 0.05, 'invariant_loss_alpha2': 0.1,
                           'invariant_loss_alpha3': 0.7, 'invariant_loss_alpha4': 0.1, 'classifier_lr': 0.001,
                           'classifier_loss_alpha': 0.001, 'environment_assign_lr': 0.001,
                           'environment_assign_loss_alpha': 1.0}

HIRL_RA_heuED_dict = {'embedding_dim': 32, 'mask1_hidden_dim': 64, 'mask2_hidden_dim': 32,
                           'predictor_hidden_dim1': 64, 'predictor_hidden_dim2': 32, 'classifier1_hidden_dim': 32,
                           'classifier2_hidden_dim': 32, 'prob_hidden_dim': 16, 'batch_size': 64, 'epoch': 10,
                           'invariant_lr': 0.05, 'invariant_loss_alpha1': 0.2, 'invariant_loss_alpha2': 0.1,
                           'invariant_loss_alpha3': 0.5, 'invariant_loss_alpha4': 0.1, 'classifier_lr': 0.001,
                           'classifier_loss_alpha': 0.001, 'environment_assign_lr': 0.001,
                           'environment_assign_loss_alpha': 0.001}

HIRL_heuUA_heuED_dict = {'embedding_dim': 64, 'mask1_hidden_dim': 64, 'mask2_hidden_dim': 64,
                              'predictor_hidden_dim1': 64, 'predictor_hidden_dim2': 32, 'classifier1_hidden_dim': 32,
                              'classifier2_hidden_dim': 32, 'prob_hidden_dim': 32, 'batch_size': 1024, 'epoch': 10,
                              'invariant_lr': 0.2, 'invariant_loss_alpha1': 0.1, 'invariant_loss_alpha2': 0.1,
                              'invariant_loss_alpha3': 0.1, 'invariant_loss_alpha4': 0.1, 'classifier_lr': 0.01,
                              'classifier_loss_alpha': 1.0}

HIRL_heuUA_heuAD_dict = {'embedding_dim': 64, 'mask1_hidden_dim': 64, 'mask2_hidden_dim': 64,
                              'predictor_hidden_dim1': 64, 'predictor_hidden_dim2': 32, 'classifier1_hidden_dim': 32,
                              'classifier2_hidden_dim': 32, 'prob_hidden_dim': 32, 'batch_size': 1024, 'epoch': 10,
                              'invariant_lr': 0.2, 'invariant_loss_alpha1': 0.1, 'invariant_loss_alpha2': 0.1,
                              'invariant_loss_alpha3': 0.1, 'invariant_loss_alpha4': 0.1, 'classifier_lr': 0.01,
                              'classifier_loss_alpha': 1.0}

HIRL_heuFA_heuAD_dict = {'embedding_dim': 64, 'mask1_hidden_dim': 64, 'mask2_hidden_dim': 64,
                              'predictor_hidden_dim1': 64, 'predictor_hidden_dim2': 32, 'classifier1_hidden_dim': 32,
                              'classifier2_hidden_dim': 32, 'prob_hidden_dim': 32, 'batch_size': 1024, 'epoch': 10,
                              'invariant_lr': 0.2, 'invariant_loss_alpha1': 0.1, 'invariant_loss_alpha2': 0.1,
                              'invariant_loss_alpha3': 0.1, 'invariant_loss_alpha4': 0.1, 'classifier_lr': 0.01,
                              'classifier_loss_alpha': 1.0}

HIRL_heuFA_heuED_dict = {'embedding_dim': 64, 'mask1_hidden_dim': 64, 'mask2_hidden_dim': 64,
                              'predictor_hidden_dim1': 64, 'predictor_hidden_dim2': 32, 'classifier1_hidden_dim': 32,
                              'classifier2_hidden_dim': 32, 'prob_hidden_dim': 32, 'batch_size': 1024, 'epoch': 10,
                              'invariant_lr': 0.2, 'invariant_loss_alpha1': 0.1, 'invariant_loss_alpha2': 0.1,
                              'invariant_loss_alpha3': 0.1, 'invariant_loss_alpha4': 0.1, 'classifier_lr': 0.01,
                              'classifier_loss_alpha': 1.0}

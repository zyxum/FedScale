init

aggregator -> trainer:  MSG_TYPE_S2C_INIT_CONFIG
- process_id
- global_model_params
- arch_params

trainer: update_model/arch()

trainer: search() / train()

trainer -> aggregator: MSG_TYPE_C2S_SEND_MODEL_TO_SERVER
- process_id
- model_params
- arch_params
- local_sample_number
- train_acc
- train_loss

aggregator: aggregate(), 

aggregator: infer(), 

aggregator: record_model_global_architecture()

aggregator -> trainer: MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT
- process_id
- model_params
- arch_params

trainer: update_model/arch()

...
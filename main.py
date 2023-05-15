import helper
import qlstm
import qlstm_lm
import lstm

print("started running main.py")

args = {}
args['embed_dims'] = 64
args['dropout_prob'] = 0.5
args['init_range'] = 0.05
args['num_layers'] = 2
args['time_steps'] = 30
args['embed_tying'] = False
args['bias'] = False
args['freq_threshold'] = 3
args['epochs'] = 20
args['learning_rate'] = 1.0
args['learning_rate_decay'] = 1.25
args['batch_size'] = 128
args['max_grad'] = 2
args['device'] = "gpu"
args['save_model'] = True
args['load_model'] = False
args['model_path'] = "/export/b17/sli136/qlstm/model"
args['topic'] = "spen"
args['path'] = "/export/b17/sli136/qlstm/data"

data_params = {k:args[k] for k in ['topic','freq_threshold', 'time_steps', 'batch_size',  'path']}
datasets = helper.init_datasets(**data_params)

# data_loaders are (train, valid, test)
data_loaders = datasets['data_loaders']

model_params = ['device', 'embed_dims', 'dropout_prob', 'init_range', 'num_layers', 'max_grad', 'bias']
model_params = {k:args[k] for k in model_params}
model_params['vocab_size'] = datasets['vocab_size']

model = qlstm_lm.Q_LSTM_Model_LM(**model_params)

# print out vocab size and perplexity on validation set
print(f"vocab size : {datasets['vocab_size']}")
px = helper.perplexity(data=data_loaders[1], model=model, batch_size=data_loaders[1].batch_size)
print("perplexity on %s dataset before training: %.3f, " % ('valid', px))


if args['load_model']:
    model.load_state_dict(torch.load(args.model_path))
else:
    training_params = ['epochs', 'learning_rate', 'learning_rate_decay', 'max_grad']
    training_params = {k:args[k] for k in training_params}
    model, perplexity_scores = helper.train(data=data_loaders, model=model, **training_params)


# now calculate perplexities for train, valid, test
for d, name in zip(data_loaders, ['train', 'valid', 'test']):
    px = perplexity(data=d, model=model, batch_size=d.batch_size)
    print("perplexity on %s dataset after training : %.3f, " % (name, px))

# save model
if args.save_model:
    torch.save(model.state_dict(), args.model_path)
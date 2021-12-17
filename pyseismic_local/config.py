config = './config.txt'
module_dir = ''
raw_data_dir = ''
with open(config, 'r') as f:
    commend = f.read()
exec(commend)

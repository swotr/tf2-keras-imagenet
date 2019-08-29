import os
import re
from absl import app
from absl import flags

def get_number(s):
    toks = re.split('[,\(\)\[\]]', s)
    for tok in toks:
        if tok != '':
            return int(tok)
    return -1

def calc_keras_flops(file):
    print(file)
    try:        
        with open(file, 'r') as f:
            mac = 0
            lines = f.readlines()
            shape = None
            prev_shape = None
            for _, line in enumerate(lines):
                toks = line.split()                
                if len(toks) < 3 or toks[0].startswith(' '):
                    continue
                if toks[2].startswith(tuple(('(None', '[(None'))):
                    if toks[3].endswith(')'):
                        # flattened
                        out_channels = get_number(toks[3])
                        shape = [out_channels]
                        params = int(toks[4])                        
                    else:
                        # image
                        out_h = get_number(toks[3])
                        out_w = get_number(toks[4])
                        out_channels = get_number(toks[5])
                        shape = [out_h, out_w, out_channels]
                        params = int(toks[6])

                if toks[0].startswith('conv2d'): 
                    kk = params // (prev_shape[2] * shape[2])
                    if kk * (prev_shape[2] * shape[2]) == params or \
                       kk * (prev_shape[2] * shape[2]) + shape[2] == params: # w bias
                        mac += shape[0] * shape[1] * prev_shape[2] * shape[2] * kk                        
                    else: # something wrong
                        print('(warning) shape does not match. skipped ', toks[0]) 
                elif toks[0].startswith('depthwise'):
                    kk = params // shape[2]
                    if kk * shape[2] == params or \
                       kk * shape[2] + shape[2]: # w bias
                        mac += shape[0] * shape[1] * shape[2] * kk
                    else:
                        # something wrong
                        print('(warning) shape does not match. skipped ', toks[0]) 
                elif toks[0].startswith('dense'):
                    mac += prev_shape[0] * shape[0]

                prev_shape = shape

            print('Total MAC (conv2d, depthwise2d, dense): {:,}'.format(mac))
            f.close()
    except Exception as e:
        print(e)

def main(argv):
    calc_keras_flops(argv[1])

if __name__ == '__main__':
    app.run(main)
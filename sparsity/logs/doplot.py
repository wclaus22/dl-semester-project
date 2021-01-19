import matplotlib.pyplot as plt 
import numpy as np 
import sys
import os
import re



def generate_images():
    files = [x for x in os.listdir('.') if x.endswith('.txt') and not x.startswith('.')]
    training_types = ['vanilla', 'sdr', 'rigl']
    for f in files:
        if '_mode_momentum' in f:
            training_type = 'momgrow'
        elif 'uniform' in f:
            training_type = 'rigl'
        elif 'zeta' in f:
            training_type = 'sdr'
        else:
            training_type = 'vanilla'
        save_images_from_file(f, training_type)



def save_images_from_file(file, training_type):
    # file = sys.argv[1] + '.txt' if not sys.argv[1].endswith('.txt') else sys.argv[1]

    val_losses = []
    val_accs = []
    with open(file) as f:
        line = f.readline()
        while(line):
            if not 'training' in line and 'validation' in line:
                
                line = line.split('validation')[1]
                data = re.findall("\d+\.\d+", line)
                data.extend([int(s) for s in line.split() if s.isdigit()])
                data = float(data[0])
        
                if 'loss' in line:
                    val_losses.append(data)
                else:
                    val_accs.append(data)
            line = f.readline()


    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(np.arange(0, len(val_losses),1), val_losses)
    ax[0].set_title(f"Loss, min = {min(val_losses)}")
    ax[1].plot(np.arange(0, len(val_accs),1), val_accs)
    ax[1].set_title(f"Accuracy, max = {max(val_accs)}")
    title_text = 'n_epochs' + file.split('.txt')[0].split('log_')[1].split('n_epochs')[1]
    fig.suptitle(f'{training_type}: {title_text}', fontsize=8)
    plt.savefig('images/fig_' + file.split('log_')[1].split('.txt')[0] + '.png')

if __name__ == '__main__':
    try:
        os.makedirs('images')
    except:
        pass
    generate_images()






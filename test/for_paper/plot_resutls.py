import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

FontSize = 18
alpha_ = 0.75

def modify_name(file_name, addition):
    index = file_name.find('.csv')
    output_line = file_name[:index] + "_" + addition + ".pdf"
    return output_line

def open_file(file_name):
    return pd.read_csv(file_name)

def plot_data(data, file_name):
    plt.plot(data['tensor'], label = 'tensor', alpha=alpha_)
    plt.plot(data['ptr_bad'], label = 'ptr cpu', alpha=alpha_)
    plt.plot(data['ptr_ok'], label = 'ptr gpu', alpha=alpha_)
    plt.xlabel("number of executions")
    meajure = "ms"
    if "host" in file_name:
        meajure = "s"
    plt.ylabel(meajure)
    plt.legend()
    plt.tight_layout()
    f_name = modify_name(file_name, "plot");
    plt.savefig(f_name)
    plt.show()    

def hist_data(data, file_name):
    plt.hist(data['tensor'], label = 'tensor', bins = 50, density=True, alpha=alpha_)
    plt.hist(data['ptr_bad'], label = 'ptr cpu', bins = 50, density=True, alpha=alpha_)
    plt.hist(data['ptr_ok'], label = 'ptr gpu',bins = 50, density=True, alpha=alpha_)
    plt.axvline(data['tensor'].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(data['ptr_bad'].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(data['ptr_ok'].mean(), color='k', linestyle='dashed', linewidth=1)
    meajure = "ms"
    if "host" in file_name:
        meajure = "s"
    plt.xlabel(meajure)    
    plt.ylabel("relative number")
    plt.legend()
    plt.tight_layout()
    f_name = modify_name(file_name, "hist");
    plt.savefig(f_name)
    plt.show(); 
    
    print("tensor/ptr_ok: {val}".format(val = data['tensor'].mean()/data['ptr_ok'].mean()) );
    print("ptr_ok/tensor: {val}".format(val = data['ptr_ok'].mean()/data['tensor'].mean()) );
    print("ptr_bad/tensor: {val}".format(val = data['ptr_bad'].mean()/data['tensor'].mean()) );
    print("ptr_bad/ptr_ok: {val}".format(val = data['ptr_bad'].mean()/data['ptr_ok'].mean()) );

def main():

    if len(sys.argv) != 2:
        print("Usage:" + sys.argv[0] + " file_name")
        return
    
    matplotlib.rcParams['font.size'] = FontSize

    file_name = sys.argv[1]
    data = open_file(file_name);
    plot_data(data, file_name)
    hist_data(data, file_name)

if __name__ == '__main__':
    main()

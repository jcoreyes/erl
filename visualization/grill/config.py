import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 15})
plt.style.use("ggplot")

output_dir = '/home/vitchyr/git/railrl/data/papers/nips2018/'
ashvin_base_dir = '/mnt/gauss1/ashvin-all-data/'
vitchyr_base_dir = '/home/vitchyr/git/railrl/data/'

def format_func(value, tick_number):
    return(str(int(value // 1000)) + 'K')

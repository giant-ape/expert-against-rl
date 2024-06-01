
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse
import matplotlib.font_manager as fm

def load_and_process(exp):
    # Load data
    files = glob.glob('{}/*/training_log.csv'.format(exp))
    all_files = [pd.read_csv(file) for file in files] # ['step'] ['success rate'] ['episodic reward']
    processed_files = []

    # Smooth data
    for datum in all_files:
        datum['reward'] = datum['episodic reward'].ewm(span=3000).mean()
        datum['success'] = datum['success rate'].ewm(span=3000).mean()
        # datum['collision'] = datum['collision'].ewm(span=3000).mean()
        # datum['off_road'] = datum['off_road'].ewm(span=3000).mean()
        # datum['safe'] = datum['safe'].ewm(span=3000).mean()
        datum = datum[datum['step'] % 50 == 0]
        processed_files.append(datum)    

    # Concatenate data
    data = pd.concat(processed_files)

    return data

# load and process data
parser = argparse.ArgumentParser()
parser.add_argument("scenario", help="scenario to plot")
parser.add_argument('metric', help='metric to plot')
parser.add_argument('step', help='metric to plot')
parser.add_argument('folder', help='metric to plot')
args = parser.parse_args()

# Load and process data for four folders
results_folders = args.folder.split(",")

results_list = []
for folder in results_folders:
    results_list.append(load_and_process(folder))

# Plot
plt.figure(figsize=(15, 10))
seaborn.set(style="whitegrid", font_scale=2, rc={"lines.linewidth": 3})
# seaborn.set_style("whitegrid", {"grid.linestyle": "--"})  # Set grid lines to dashed

for i, results in enumerate(results_list):
    if i == 0:
        seaborn.lineplot(data=results, x='step', y=f'{args.metric}', err_style='band',label='sac')
    elif i == 1:
        seaborn.lineplot(data=results, x='step', y=f'{args.metric}', err_style='band',label='policy_constraint')
    elif i == 2:
        seaborn.lineplot(data=results, x='step', y=f'{args.metric}', err_style='band', label='value_penalty')
    elif i == 3:
        seaborn.lineplot(data=results, x='step', y=f'{args.metric}', err_style='band', label='expert_against')

axes = plt.gca()

axes.set_xlim([0, int(args.step)])
axes.set_xlabel('Step')

if args.metric == 'success':
    axes.set_ylim([0, 1])
    axes.set_ylabel('Average success rate')
# elif args.metric == 'collision':
#     axes.set_ylim([0, 1])
#     axes.set_ylabel('Average episode no collision rate')
# elif args.metric == 'off_road':
#     axes.set_ylim([0, 1])
#     axes.set_ylabel('Average episode no off_road rate')
# elif args.metric == 'safe':
#     axes.set_ylim([0, 1])
#     axes.set_ylabel('Average episode safe rate')
else:
    raise Exception('Undefined metric!')
    
axes.set_title(f'{args.scenario}')

plt.tight_layout()
gif_path = './'
plt.savefig('{}/{}_{}_a.png'.format(gif_path,args.scenario,args.metric), dpi=500)


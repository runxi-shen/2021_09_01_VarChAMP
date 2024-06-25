import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def rescale(df):
    df -= df.min()
    df /= df.max()
    return df

def colorcode(x):
    if x == 'Pathogenic':
        return 'red'
    elif x == 'Benign':
        return 'green'
    elif x == 'Conflicting':
        return 'gray'
    else: return 'blue'
        
def plot_figure(df_chosen_var, color_dict, chosen_gene):
    plt.figure(figsize=[4,10])
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(df_chosen_var, cmap=cmap, cbar_kws={"shrink": .5}, annot=True)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='major', labelsize=10)
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(color_dict[tick_label.get_text()])
    ax.set_title(chosen_gene)
    plt.savefig(f'fig/{chosen_gene}.png', format='png', bbox_inches='tight')
    
def generate_figure(df_merged, chosen_gene):
    df_chosen = df_merged[df_merged['Gene']==chosen_gene].reset_index(drop=True)
    
    color_dict = dict(zip(df_chosen['Variant'], df_chosen['clinical_significance'].apply(lambda x: colorcode(x))))

    df_chosen_var = df_chosen[['Variant', 'F1_Score', 'z_abundance']].copy()
    df_chosen_var.sort_values(by=['F1_Score'], inplace=True, ascending=False)
    df_chosen_var.index = df_chosen_var['Variant']
    df_chosen_var.drop ('Variant', axis=1, inplace=True)
    df_chosen_var.dropna(inplace=True)
    if df_chosen_var.size==0:
        return
    
    df_chosen_var['z_abundance'] = rescale(df_chosen_var['z_abundance'])
    df_chosen_var['F1_Score'] = rescale(df_chosen_var['F1_Score'])

    plot_figure(df_chosen_var, color_dict, chosen_gene)
    
def main():
    ppi_path = 'abundance_ppi_his3_edgotyping.tsv'
    cp_path = '../../../results/Run7/2023_05_30_B1A1R1_protein_channel_f1score_agg.csv'

    df_cp = pd.read_csv(cp_path)
    df_cp['Variant'] = df_cp['Variant'].apply(lambda x: x.split()[1])
    
    df_ppi = pd.read_csv(ppi_path, sep='\t', header=0)
    
    df_merged = df_cp.merge(df_ppi, left_on=['Gene', 'Variant'], right_on=['symbol', 'aa_change'], how='inner')

    for gene in tqdm(df_merged['Gene'].unique()):
        generate_figure(df_merged, gene)
        
main()
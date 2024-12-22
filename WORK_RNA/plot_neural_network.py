import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

def place_committee(n_input, n_members):
    """Determina posições do comitê de modelos lineares para plotar.
    
    Inputs:
        n_input: número de entradas do modelo
        n_members: número de membros do comitê
        
    Outputs:
        x_input: posição x das entradas
        y_input: posição y das entradas
        l_input: lado do quadrado que representa as entradas
        color_input: cor das entradas
        x_committee: posição x dos membros do comitê
        y_committee: posição y dos membros do comitê
        r_committee: raio do círculo que representa os membros do comitê
        color_committee: cor dos membros do comitê
        x_output: posição x da saída
        y_output: posição y da saída
        r_output: raio do círculo que representa a saída
        color_output: cor da saída
    """

    ##########################################
    ### Posicionamento das entradas do modelo
    ##########################################

    l_input = 0.02 # Lado do quadrado que representa as entradas (arbitrário)
    x_input = 1/10 # Posição x das entradas (arbitrário)
    ymin_input = 1/5 # Posição y mínima das entradas (arbitrário)
    ymax_input = 1-1/5 # Posição y máxima das entradas (arbitrário)
    y_input = np.linspace(ymin_input, ymax_input, n_input) # Posição y das entradas
    color_input = 'slategrey'
    #
    # Se o número de entradas for muito grande pra se usar o tamanho usual
    # de quadrados, diminua o tamanho dos quadrados
    while l_input*n_input >= (ymax_input - ymin_input):
        l_input = l_input/2

    #########################################
    ### Posicionamento dos membros do comitê
    #########################################

    m = n_members # Número de membros do comitê
    x_committee = 1/2 # Posição x dos membros do comitê (arbitrário)
    ymin_committee = 1/10 # Posição y mínima dos membros do comitê (arbitrário)
    ymax_committee = 1-1/10 # Posição y máxima dos membros do comitê (arbitrário)
    min_distance = 1/20 # Distância mínima entre os membros do comitê (arbitrário)
    color_committee = 'lightsteelblue' 
    r_committee = 1/20 # Raio do círculo que representa os membros do comitê (arbitrário) 
    #
    # Se o número de neurônios for muito grande pra se usar o valor usual do raio,
    # ajuste o tamanho do raio e a distância mínima entre os membros
    if min_distance*(m-1) > ymax_committee - ymin_committee:
        min_distance = 0
    if 2*r_committee*m + min_distance*(m-1) >= ymax_committee - ymin_committee:
        r_committee = (ymax_committee - ymin_committee - min_distance*(m-1))/(2*m)
    #
    y_committee = np.linspace(ymin_committee+r_committee, ymax_committee-r_committee, m)  # Posição y dos membros do comitê

    ############################
    ### Posicionamento da saída
    ############################

    r_output = 1/18 # Raio do círculo que representa a saída (arbitrário)
    x_output = 1 - (r_output + 1/10) # Posição x da saída (arbitrário)
    y_output = 1/2 # Posição y da saída (arbitrário)
    color_output = 'lightsteelblue'

    return(x_input, y_input, l_input, color_input,
           x_committee, y_committee, r_committee, color_committee,
           x_output, y_output, r_output, color_output)
    
    
def plot_committee(n_inputs, n_members,
                   x_input, y_input, l_input, color_input,
                   x_committee, y_committee, r_committee, color_committee,
                   x_output, y_output, r_output, color_output,
                   ax=None, x_fig=8, y_fig=8):
    """Plota um comitê de modelos lineares.
    
    Inputs:
        n_inputs: número de entradas do modelo
        n_members: número de membros do comitê
        x_input: posição x das entradas
        y_input: posição y das entradas
        l_input: lado do quadrado que representa as entradas
        color_input: cor das entradas
        x_committee: posição x dos membros do comitê
        y_committee: posição y dos membros do comitê
        r_committee: raio do círculo que representa os membros do comitê
        color_committee: cor dos membros do comitê
        x_output: posição x da saída
        y_output: posição y da saída
        r_output: raio do círculo que representa a saída
        color_output: cor da saída
        ax (opcional): eixo onde o comitê será plotado. Se não for fornecido, um novo eixo será criado
        x_fig (opcional): largura da figura. Default = 8.
        y_fig (opcional): altura da figura. Default = 8.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(x_fig, y_fig))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.autoscale(False)
        ax.grid(False)
    
    # Entradas são quadrados
    for j in range(n_inputs):
        square = plt.Rectangle((x_input - l_input/2, y_input[j] - l_input/2), l_input, l_input,
                               facecolor=color_input, edgecolor='black', linewidth=0.5,
                               zorder=3)
        ax.add_patch(square)
    
    # Membros do comitê
    for j in range(n_members):
        circle = plt.Circle((x_committee, y_committee[j]), r_committee,
                            facecolor=color_committee, edgecolor='black', linewidth=0.5,
                            zorder=3)
        ax.add_patch(circle)
        
    # Saída
    circle = plt.Circle((x_output, y_output), r_output,
                        facecolor=color_output, edgecolor='black', linewidth=0.5,
                        zorder=3)
    ax.text(x_output, y_output, r'$\Sigma$', fontsize=5*x_fig, color='black', ha='center', va='center_baseline', zorder=4)
    ax.add_patch(circle)
    
    # Conecta cada entrada ao comitê
    for j in range(n_inputs):
        for k in range(n_members):
            ax.plot([x_input, x_committee], [y_input[j], y_committee[k]], color='k', alpha=0.5, linewidth=0.5)
            
    # Conecta cada membro do comitê à saída
    for j in range(n_members):
        ax.plot([x_committee, x_output], [y_committee[j], y_output], color='k', alpha=0.5, linewidth=0.5)
    
    # Seta para indicar a saída
    ax.annotate('', xy=(x_output, y_output), 
                xytext=(x_output + 0.095, y_output),
                arrowprops=dict(facecolor='black', arrowstyle='<-'),
                zorder=1,
                fontsize=2*x_fig)
    
    ax.set_xticks([])
    ax.set_yticks([])


def plot_nn(n_members : int):
    
    n_input = 4
    #n_members = 3
    
    (x_input, y_input, l_input, color_input,
    x_committee, y_committee, r_committee, color_committee,
    x_output, y_output, r_output, color_output) = place_committee(n_input, n_members)
    
    plot_committee(n_input, n_members,
                   x_input, y_input, l_input, color_input,
                   x_committee, y_committee, r_committee, color_committee,
                   x_output, y_output, r_output, color_output)
    
    plt.show()
